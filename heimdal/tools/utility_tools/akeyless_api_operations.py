import base64, json, os, pprint, akeyless, xml.etree.ElementTree as ET, logging, requests
from langchain.tools import tool
from typing import Tuple, Union
import akeyless
from akeyless.rest import ApiException
from akeyless.models import CreateAuthMethodAWSIAMOutput, CreateAuthMethodGCPOutput , CreateAuthMethodAzureADOutput, CreateAuthMethodOutput, ValidateTokenOutput
from akeyless_cloud_id import CloudId

# Initialize the CloudId generator
cloud_id_generator = CloudId()

# Set the Akeyless API endpoint
akeyless_api_endpoint = os.getenv('AKEYLESS_API_GW_URL', "https://api.akeyless.io")

# Configure the Akeyless API client
configuration = akeyless.Configuration(host=akeyless_api_endpoint)

# Create an instance of the Akeyless API client
api_client = akeyless.ApiClient(configuration)

# Create an instance of the Akeyless V2 API
api = akeyless.V2Api(api_client)

# Retrieve the Akeyless token from the environment variables


def extract_aws_account_id(cloud_id_b64: str) -> str:
    """
    This function extracts the AWS account ID from the base64 encoded cloud_id.

    Args:
        cloud_id_b64 (str): The base64 encoded cloud_id.

    Returns:
        str: The AWS account ID.
    """
    logging.info("Decoding the base64 cloud_id to get the JSON string.")
    cloud_id_json_str: str = base64.b64decode(cloud_id_b64).decode('utf-8')
    cloud_id_data: dict = json.loads(cloud_id_json_str)

    logging.info("Decoding the sts_request_headers.")
    headers_json_str: str = base64.b64decode(cloud_id_data['sts_request_headers']).decode('utf-8')
    headers: dict = json.loads(headers_json_str)

    # Convert headers from list to dictionary format if necessary
    headers = {k: v[0] if isinstance(v, list) else v for k, v in headers.items()}
    
    # Prepare the request
    url: str = 'https://sts.amazonaws.com'
    body: str = 'Action=GetCallerIdentity&Version=2011-06-15'

    logging.info("Making the request to AWS STS.")
    response = requests.post(url, data=body, headers=headers)

    # Check for HTTP errors
    response.raise_for_status()

    logging.info("Parsing the response to extract the AWS account ID.")
    root = ET.fromstring(response.text)
    namespace: dict = {'sts': 'https://sts.amazonaws.com/doc/2011-06-15/'}

    # Extract the account ID from the XML
    account_id: str = root.find('.//sts:Arn', namespace).text.split(':')[4]

    logging.debug(f"AWS Account ID: {account_id}")
    return account_id

def extract_gcp_project_id(cloud_id_b64: str) -> str:
    """
    This function extracts the GCP project ID from the base64 encoded cloud_id.

    Args:
        cloud_id_b64 (str): The base64 encoded cloud_id.

    Returns:
        str: The GCP project ID.
    """
    # Decode the base64 cloud_id to get the JSON string
    logging.info("Decoding the base64 cloud_id to get the JSON string.")
    logging.debug("JWT token read successfully.")
    cloud_id_json_str: str = base64.b64decode(cloud_id_b64).decode('utf-8')
    # Split the JWT token which is in 'header.payload.signature' format
    payload: str = cloud_id_json_str.split('.')[1]
    # Correct padding for Base64 decoding
    payload += '=' * (-len(payload) % 4)
    payload_json: str = base64.urlsafe_b64decode(payload).decode('utf-8')
    payload_data: dict = json.loads(payload_json)

    # Extract the GCP project ID from the decoded JSON data
    logging.info("Extracting the GCP project ID from the decoded JSON data.")
    try:
        project_id: str = payload_data["google"]["compute_engine"]["project_id"]
    except KeyError:
        logging.warning("Failed to extract project ID from the decoded JSON data.")
        logging.info("Checking the 'email' attribute to attempt to extract the project ID.")
        try:
            project_id: str = payload_data["email"].split('@')[1].split('.')[0]
        except Exception as e:
            logging.error(f"Failed to extract project ID from the 'email' attribute. Error: {e}")
            project_id: str = None

    logging.debug(f"GCP Project ID: {project_id}")
    return project_id

def extract_azure_tenant_id(cloud_id_b64: str) -> str:
    """
    This function extracts the Azure tenant ID from the base64 encoded cloud_id.

    Args:
        cloud_id_b64 (str): The base64 encoded cloud_id.

    Returns:
        str: The Azure tenant ID.
    """
    # Decode the base64 cloud_id to get the JSON string
    logging.info("Decoding the base64 cloud_id to get the JSON string.")
    cloud_id_json_str: str = base64.b64decode(cloud_id_b64).decode('utf-8')
    cloud_id_data: dict = json.loads(cloud_id_json_str)

    # Extract the Azure tenant ID from the decoded JSON data
    logging.info("Extracting the Azure tenant ID from the decoded JSON data.")
    azure_tenant_id: str = cloud_id_data.get('tenant_id')

    logging.debug(f"Azure Tenant ID: {azure_tenant_id}")
    return azure_tenant_id


def _create_auth_method_name(cloud_service_provider: str, auth_method_name_template: str = '/Heimdal AI Created {cloud_service_provider} Auth Method') -> str:
    logging.info(f"Creating Akeyless cloud authentication method for {cloud_service_provider}")
    auth_method_name: str = auth_method_name_template.format(cloud_service_provider=cloud_service_provider)
    return auth_method_name


def create_aws_cloud_auth_method() -> str:
    """
    This function creates an AWS IAM authentication method in Akeyless.

    Returns:
        str: The access ID of the created authentication method.
    """
    # Get the Akeyless token from the environment variables
    token: str = os.getenv('AKEYLESS_TOKEN')
    # Create the authentication method name for AWS
    auth_method_name: str = _create_auth_method_name("AWS")
    logging.debug(f"Auth method name: {auth_method_name}")
    # Generate the cloud ID for AWS
    cloud_id: str = cloud_id_generator.generate()
    # Extract the AWS account ID from the cloud ID
    aws_account_id: str = extract_aws_account_id(cloud_id)
    logging.debug(f"AWS Account ID: {aws_account_id}")
    # Create the request body for the Akeyless API
    body: akeyless.CreateAuthMethodAWSIAM = akeyless.CreateAuthMethodAWSIAM(token=token, bound_aws_account_id=aws_account_id, name=auth_method_name)
    api_response: CreateAuthMethodAWSIAMOutput = None

    try:
        # Call the Akeyless API to create the AWS IAM authentication method
        api_response = api.create_auth_method_awsiam(body)
        logging.debug(api_response)
        return api_response.access_id
    except ApiException as e:
        # Log an error if the API call fails
        logging.error("Exception when calling V2Api->create_auth_method_awsiam: %s\n" % e)
        raise


def create_azure_cloud_auth_method() -> str:
    """
    This function creates an Azure AD authentication method in Akeyless.

    Returns:
        str: The access ID of the created authentication method.
    """
    # Get the Akeyless token from the environment variables
    token = os.getenv('AKEYLESS_TOKEN')
    # Create the authentication method name for Azure
    auth_method_name = _create_auth_method_name("AZURE")
    logging.debug(f"Auth method name: {auth_method_name}")
    # Generate the cloud ID for Azure
    cloud_id: str = cloud_id_generator.generateAzure()
    # Extract the Azure tenant ID from the cloud ID
    azure_tenant_id: str = extract_azure_tenant_id(cloud_id)
    logging.debug(f"Azure Tenant ID: {azure_tenant_id}")
    # Create the request body for the Akeyless API
    body: akeyless.CreateAuthMethodAzureAD = akeyless.CreateAuthMethodAzureAD(token=token, bound_tenant_id=azure_tenant_id, name=auth_method_name)
    api_response: CreateAuthMethodAzureADOutput = None

    try:
        # Call the Akeyless API to create the Azure AD authentication method
        api_response = api.create_auth_method_azure(body)
        logging.debug(api_response)
        return api_response.access_id
    except ApiException as e:
        # Log an error if the API call fails
        logging.error("Exception when calling V2Api->create_auth_method_azure: %s\n" % e)
        raise


def create_gcp_cloud_auth_method() -> str:
    """
    This function creates a GCP authentication method in Akeyless.

    Returns:
        str: The access ID of the created authentication method.
    """
    # Get the Akeyless token from the environment variables
    token = os.getenv('AKEYLESS_TOKEN')
    # Create the authentication method name for GCP
    auth_method_name = _create_auth_method_name("GCP")
    logging.debug(f"Auth method name: {auth_method_name}")
    # Generate the cloud ID for GCP
    cloud_id: str = cloud_id_generator.generateGcp()
    # Extract the GCP project ID from the cloud ID
    gcp_project_id: str = extract_gcp_project_id(cloud_id)
    logging.debug(f"GCP Project ID: {gcp_project_id}")
    # Create the request body for the Akeyless API
    body: akeyless.CreateAuthMethodGCP = akeyless.CreateAuthMethodGCP(token=token, bound_projects=[gcp_project_id], name=auth_method_name, type="gce")
    
    logging.debug(f"Creating Akeyless GCP authentication method with the following parameters: {body.to_str()}")

    try:
        # Call the Akeyless API to create the GCP authentication method
        api_response: CreateAuthMethodGCPOutput = api.create_auth_method_gcp(body)
        logging.debug(api_response)
        return api_response.access_id
    except ApiException as e:
        # Log an error if the API call fails
        logging.error("Exception when calling V2Api->create_auth_method_gcp: %s\n" % e)
        raise


def create_akeyless_api_key_auth_method() -> str:
    """
    This function creates an Akeyless API Key authentication method.

    Args:
        name (str): The name of the authentication method.

    Returns:
        Tuple[str, CreateAuthMethodOutput]: The access ID of the created authentication method and the API response.
    """
    token = os.getenv('AKEYLESS_TOKEN')
    auth_method_name = _create_auth_method_name("API Key")
    logging.info(f"Creating Akeyless API Key Auth Method: {auth_method_name}")

    # Create the body for the API request
    body: akeyless.CreateAuthMethod = akeyless.CreateAuthMethod(token=token, name=auth_method_name)

    try:
        # Call the Akeyless API to create the authentication method
        api_response: CreateAuthMethodOutput = api.create_auth_method_api_key(body)
        logging.debug(api_response)

        # Extract the access ID from the API response
        auth_method_access_id = api_response.access_id
        logging.info(f"Created Akeyless API Key Auth Method with Access ID: {auth_method_access_id}")

        return auth_method_access_id
    except ApiException as e:
        logging.error(f"Exception when calling V2Api->create_auth_method_api_key: {e}")
        raise


def create_akeyless_cloud_auth_method(cloud_service_provider: str) -> Tuple[str, Union[CreateAuthMethodAWSIAMOutput, CreateAuthMethodGCPOutput, CreateAuthMethodAzureADOutput]]:
    """
    This function creates an Akeyless cloud authentication method based on the cloud service provider.

    Args:
        cloud_service_provider (str): The cloud service provider ('aws', 'gcp', 'azure').

    Returns:
        Tuple[str, Union[CreateAuthMethodAWSIAMOutput, CreateAuthMethodGCPOutput, CreateAuthMethodAzureADOutput]]: The access ID of the created cloud authentication method and the API response.
    """
    token = os.getenv('AKEYLESS_TOKEN')
    logging.info(f"Creating Akeyless cloud authentication method for {cloud_service_provider.upper()}")
    cloud_auth_method_access_id: str = None
    cloud_id: str = cloud_id_generator.generate()
    auth_method_name: str = f'/Heimdal AI Created {cloud_service_provider.upper()} Auth Method'

    if cloud_service_provider == 'aws':
        aws_account_id: str = extract_aws_account_id(cloud_id)
        body: akeyless.CreateAuthMethodAWSIAM = akeyless.CreateAuthMethodAWSIAM()
        body.token = token
        body.bound_aws_account_id = aws_account_id
        body.name = auth_method_name

        try:
            api_response: CreateAuthMethodAWSIAMOutput = api.create_auth_method_awsiam(body)
            logging.debug(api_response)
            cloud_auth_method_access_id = api_response.access_id
            return cloud_auth_method_access_id, api_response
        except ApiException as e:
            logging.error("Exception when calling V2Api->create_auth_method_awsiam: %s\n" % e)
            raise
    elif cloud_service_provider == 'gcp':
        gcp_project_id: str = extract_gcp_project_id(cloud_id)
        body: akeyless.CreateAuthMethodGCP = akeyless.CreateAuthMethodGCP()
        body.token = token
        body.bound_gcp_project_id = gcp_project_id
        body.name = auth_method_name

        try:
            api_response: CreateAuthMethodGCPOutput = api.create_auth_method_gcp(body)
            logging.debug(api_response)
            cloud_auth_method_access_id = api_response.access_id
            return cloud_auth_method_access_id, api_response
        except ApiException as e:
            logging.error("Exception when calling V2Api->create_auth_method_gcp: %s\n" % e)
            raise
    elif cloud_service_provider == 'azure':
        azure_tenant_id: str = extract_azure_tenant_id(cloud_id)
        body: akeyless.CreateAuthMethodAzure = akeyless.CreateAuthMethodAzure()
        body.token = token
        body.bound_azure_tenant_id = azure_tenant_id
        body.name = auth_method_name

        try:
            api_response: CreateAuthMethodAzureADOutput = api.create_auth_method_azure(body)
            logging.debug(api_response)
            cloud_auth_method_access_id = api_response.access_id
            return cloud_auth_method_access_id, api_response
        except ApiException as e:
            logging.error("Exception when calling V2Api->create_auth_method_azure: %s\n" % e)
            raise
    
    return cloud_auth_method_access_id, api_response




def validate_akeyless_token(token) -> ValidateTokenOutput:
    """
    This function validates the Akeyless token.

    Args:
        token (str): The Akeyless token to be validated.

    Returns:
        str: The validation result.
    """
    logging.info("Validating Akeyless token.")
    
    # Assuming there is a validation API endpoint or method
    try:
        # Call the Akeyless API to validate the token
        validation_result: ValidateTokenOutput = api.validate_token(token)
        logging.debug(f"Validation result: {validation_result}")

        if validation_result == 'valid':
            logging.info("Akeyless token is valid.")
        else:
            logging.info("Akeyless token is invalid.")

        return validation_result
    except ApiException as e:
        logging.error(f"Exception when calling V2Api->validate_token: {e}")
        raise


def 
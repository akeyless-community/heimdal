import base64, json, os, akeyless, xml.etree.ElementTree as ET, logging, requests
from typing import List, Tuple, Union
import akeyless
from akeyless.rest import ApiException
from akeyless.models import CreateAuthMethodAWSIAMOutput, CreateAuthMethodGCPOutput , CreateAuthMethodAzureADOutput, CreateAuthMethodOutput, ValidateTokenOutput, ListAuthMethodsOutput, ListAuthMethods, AuthMethod
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


async def extract_aws_account_id(cloud_id_b64: str) -> str:
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

async def extract_gcp_project_id(cloud_id_b64: str) -> Tuple[str, str]:
    """
    This function extracts the GCP project ID or the full GCP service account email address from the base64 encoded cloud_id and identifies if the project ID or the service account email was extracted through GCE or GKE workload identity (iam).
    It supports both GCE metadata and GKE workload identity formats.
    In the case of GKE workload identity, it will return the full Google service account email used.

    Args:
        cloud_id_b64 (str): The base64 encoded cloud_id.

    Returns:
        Tuple[str, str]: The GCP project ID or the full GCP service account email address and the source through which it was extracted (gce or iam).
    """
    source = ""
    extracted_value = ""
    try:
        # Decode the base64 cloud_id to get the JSON string
        logging.info("Decoding the base64 cloud_id to get the JSON string.")
        cloud_id_json_str: str = base64.b64decode(cloud_id_b64).decode('utf-8')
        logging.debug(f"Decoded cloud_id JSON string: {cloud_id_json_str}")

        # Split the JWT token which is in 'header.payload.signature' format
        payload: str = cloud_id_json_str.split('.')[1]
        logging.debug(f"Extracted payload from JWT: {payload}")

        # Correct padding for Base64 decoding
        payload += '=' * (-len(payload) % 4)
        logging.debug(f"Corrected payload padding for Base64 decoding.")

        payload_json: str = base64.urlsafe_b64decode(payload).decode('utf-8')
        payload_data: dict = json.loads(payload_json)
        logging.debug(f"Decoded payload JSON data: {payload_data}")

        # Extract the GCP project ID or the full GCP service account email address from the decoded JSON data
        logging.info("Extracting the GCP project ID or the full GCP service account email address from the decoded JSON data.")
        if "google" in payload_data:
            extracted_value = payload_data["google"]["compute_engine"]["project_id"]
            source = "gce"
            logging.debug(f"Extracted GCP project ID from 'google' key: {extracted_value}")
        elif "email" in payload_data:
            extracted_value = payload_data["email"]
            source = "iam"
            logging.debug(f"Extracted full GCP service account email address from 'email' key: {extracted_value}")
        else:
            logging.warning("Unable to find GCP project ID or the full GCP service account email address in the provided cloud_id.")

        logging.info(f"Extracted value: {extracted_value} extracted through {source}")
        return extracted_value, source
    except Exception as e:
        logging.error(f"Error extracting GCP project ID or the full GCP service account email address: {e}")
        raise

async def extract_azure_tenant_id(cloud_id_b64: str) -> str:
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


async def _create_auth_method_name(cloud_service_provider: str, auth_method_name_template: str = '/Heimdal AI Created {cloud_service_provider} Auth Method') -> str:
    """
    Generates a formatted authentication method name for a given cloud service provider.

    This function takes the name of a cloud service provider (e.g., AWS, Azure, GCP) and a template string,
    then formats the template with the cloud service provider's name to generate a unique authentication method name.

    Args:
        cloud_service_provider (str): The name of the cloud service provider for which to create the authentication method name.
        auth_method_name_template (str): A template string for the authentication method name. Defaults to '/Heimdal AI Created {cloud_service_provider} Auth Method'.

    Returns:
        str: The formatted authentication method name.

    Example:
        >>> _create_auth_method_name('AWS')
        '/Heimdal AI Created AWS Auth Method'
    """
    logging.info(f"Initiating the creation of an Akeyless cloud authentication method name for {cloud_service_provider}.")
    auth_method_name: str = auth_method_name_template.format(cloud_service_provider=cloud_service_provider)
    logging.debug(f"Generated authentication method name: {auth_method_name}")
    return auth_method_name


async def create_aws_cloud_auth_method() -> str:
    """
    This function creates an AWS IAM authentication method in Akeyless.

    Returns:
        str: The access ID of the created authentication method.
    """
    # Get the Akeyless token from the environment variables
    token: str = os.getenv('AKEYLESS_TOKEN')
    # Create the authentication method name for AWS
    auth_method_name: str = await _create_auth_method_name("AWS")
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


async def create_azure_cloud_auth_method() -> str:
    """
    This function creates an Azure AD authentication method in Akeyless.

    Returns:
        str: The access ID of the created authentication method.
    """
    # Get the Akeyless token from the environment variables
    token = os.getenv('AKEYLESS_TOKEN')
    # Create the authentication method name for Azure
    auth_method_name = await _create_auth_method_name("AZURE")
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


async def create_gcp_cloud_auth_method() -> str:
    """
    This function creates a GCP authentication method in Akeyless.

    Returns:
        str: The access ID of the created authentication method.
    """
    # Get the Akeyless token from the environment variables
    token = os.getenv('AKEYLESS_TOKEN')
    # Create the authentication method name for GCP
    auth_method_name = await _create_auth_method_name("GCP")
    logging.debug(f"Auth method name: {auth_method_name}")
    
    # Generate the cloud ID for GCP
    cloud_id: str = cloud_id_generator.generateGcp()
    logging.debug(f"Cloud ID for GCP: {cloud_id}")
    
    # Extract the GCP project ID from the cloud ID
    gcp_project_id: str
    extraction_source: str
    extracted_value, extraction_source = await extract_gcp_project_id(cloud_id)
    logging.debug(f"GCP ID: {extracted_value}")
    logging.debug(f"GCP Project ID extraction source: {extraction_source}")
    
    # Create the request body for the Akeyless API based on the extraction source
    if extraction_source == "gce":
        body: akeyless.CreateAuthMethodGCP = akeyless.CreateAuthMethodGCP(
            token=token,
            bound_projects=[extracted_value],
            name=auth_method_name,
            type=extraction_source
        )
    elif extraction_source == "iam":
        try:
            # Attempt to extract gcp project ID from gcp service account email
            gcp_project_id = extracted_value.split('@')[1].split('.')[0]
            logging.debug(f"Extracted GCP project ID from GCP service account email: {gcp_project_id}")
        except IndexError as e:
            logging.error(f"Failed to extract GCP project ID from service account email: {extracted_value}. Error: {e}")
            raise ValueError(f"Invalid GCP service account email format: {extracted_value}") from e

        body: akeyless.CreateAuthMethodGCP = akeyless.CreateAuthMethodGCP(
            token=token,
            bound_projects=[gcp_project_id],
            bound_service_accounts=[extracted_value],
            name=auth_method_name,
            type=extraction_source
        )
    
    logging.debug(f"Creating Akeyless GCP authentication method with the following parameters: {body.to_str()}")

    try:
        # Call the Akeyless API to create the GCP authentication method
        api_response: CreateAuthMethodGCPOutput = api.create_auth_method_gcp(body)
        logging.debug(api_response)
        return api_response.access_id
    except ApiException as e:
        # Log an error if the API call fails
        logging.error(f"Exception when calling V2Api->create_auth_method_gcp: {e}")
        raise


async def create_akeyless_api_key_auth_method() -> str:
    """
    This function creates an Akeyless API Key authentication method.

    Args:
        name (str): The name of the authentication method.

    Returns:
        Tuple[str, CreateAuthMethodOutput]: The access ID of the created authentication method and the API response.
    """
    token = os.getenv('AKEYLESS_TOKEN')
    auth_method_name = await _create_auth_method_name("API Key")
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


async def create_akeyless_cloud_auth_method(cloud_service_provider: str) -> Tuple[str, Union[CreateAuthMethodAWSIAMOutput, CreateAuthMethodGCPOutput, CreateAuthMethodAzureADOutput]]:
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
        aws_account_id: str = await extract_aws_account_id(cloud_id)
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
        gcp_project_id: str = await extract_gcp_project_id(cloud_id)
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
        azure_tenant_id: str = await extract_azure_tenant_id(cloud_id)
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



async def validate_akeyless_token(token: str) -> ValidateTokenOutput:
    """
    This function validates the Akeyless token and returns the details of the validation result.

    Args:
        token (str): The Akeyless token to be validated.

    Returns:
        ValidateTokenOutput: An object containing the validation result details, including:
            - expiration (str): The expiration time of the token.
            - is_valid (bool): A boolean indicating whether the token is valid.
            - reason (str): The reason for the token's validation status.
    """
    logging.info("Validating Akeyless token.")
    
    body: akeyless.ValidateToken = akeyless.ValidateToken()
    body.token = token
    
    try:
        # Call the Akeyless API to validate the token
        validation_result: ValidateTokenOutput = api.validate_token(body)
        logging.debug(f"Validation result: {validation_result}")

        if validation_result.is_valid:
            logging.info("Akeyless token is valid.")
        else:
            logging.info(f"Akeyless token is invalid. Reason: {validation_result.reason}")

        return validation_result
    except ApiException as e:
        logging.error(f"Exception when calling V2Api->validate_token: {e}")
        raise



# This function checks if an Akeyless authentication method with a specific name already exists.
# It uses the list_auth_methods API to filter for the specific name and then checks if the name exists in the returned list.
def check_if_akeyless_auth_method_exists_from_list_auth_methods(auth_method_name: str) -> bool:
    """
    This function checks if an Akeyless authentication method with a specific name already exists.
    It uses the list_auth_methods API to filter for the specific name and then checks if the name exists in the returned list.

    Args:
        auth_method_name (str): The name of the Akeyless authentication method to check.

    Returns:
        bool: True if the authentication method exists, False otherwise.
    """
    logging.info(f"Checking if Akeyless authentication method with name {auth_method_name} exists.")
    token = os.getenv('AKEYLESS_TOKEN')
    list_auth_methods_body: ListAuthMethods = akeyless.ListAuthMethods()
    list_auth_methods_body.token = token
    if not auth_method_name.startswith('/'):
        auth_method_name = f'/{auth_method_name}'
    list_auth_methods_body.filter = auth_method_name
    try:
        list_auth_methods_output: ListAuthMethodsOutput = api.list_auth_methods(list_auth_methods_body)
        logging.debug(list_auth_methods_output)
        auth_methods: List[AuthMethod] = list_auth_methods_output.auth_methods
        for auth_method in auth_methods:
            if auth_method.auth_method_name == auth_method_name:
                return True
        return False
    except ApiException as e:
        logging.error(f"Exception when calling V2Api->list_auth_methods: {e}")
        raise



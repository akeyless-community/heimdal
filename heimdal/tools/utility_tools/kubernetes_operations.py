import base64, json, logging, requests, subprocess, yaml, tempfile, os
from langchain_core.tools import ToolException, StructuredTool
from langchain.tools import tool
from typing import List, Tuple, Union
from pyhelm3 import Client
from kubernetes import client, config, config as k8s_config

# Create a logger object
logger = logging.getLogger(__name__)


async def generate_k8s_secret_from_literal_values(secret_name: str, namespace: str, literal_values: dict) -> None:
    """
    Generates a Kubernetes secret from literal values.

    Args:
        secret_name (str): The name of the secret to be created.
        namespace (str): The namespace in which the secret will be created.
        literal_values (dict): A dictionary of key-value pairs where the key is the name of the secret data and the value is the literal data to be stored.
    """
    logger.info(f"Generating Kubernetes secret '{secret_name}' in namespace '{namespace}'...")
    try:
        # Create a Kubernetes client
        api_instance = client.CoreV1Api()

        # Create a dictionary of data to be stored in the secret
        data = {key: value for key, value in literal_values.items()}

        # Create the secret object
        body = client.V1Secret(
            api_version="v1",
            data=data,
            kind="Secret",
            metadata=client.V1ObjectMeta(name=secret_name)
        )

        # Create the secret
        api_instance.create_namespaced_secret(namespace, body)
        logger.info("Kubernetes secret generated successfully.")
    except Exception as e:
        logger.error(f"Error generating Kubernetes secret: {e}")
        raise


async def generate_k8s_secret_from_literal_values(secret_name: str, namespace: str, literal_values: dict) -> str:
    """
    Generates a Kubernetes secret from literal values.

    Args:
        secret_name (str): The name of the secret to be created.
        namespace (str): The namespace in which the secret will be created.
        literal_values (dict): A dictionary of key-value pairs where the key is the name of the secret data and the value is the literal data to be stored.

    Returns:
        str: The result of the Kubernetes secret generation.
    """
    logger.info(f"Generating Kubernetes secret '{secret_name}' in namespace '{namespace}'...")
    try:
        # Create a Kubernetes client
        api_instance = client.CoreV1Api()

        # Create a dictionary of data to be stored in the secret
        data = {key: value for key, value in literal_values.items()}

        # Create the secret object
        body = client.V1Secret(
            api_version="v1",
            data=data,
            kind="Secret",
            metadata=client.V1ObjectMeta(name=secret_name)
        )

        # Create the secret
        api_response = api_instance.create_namespaced_secret(namespace, body)
        logger.info("Kubernetes secret generated successfully.")
        return json.dumps(api_response)
    except Exception as e:
        logger.error(f"Error generating Kubernetes secret: {e}")
        raise

async def load_kubernetes_config() -> Tuple[str, client.Configuration]:
    """
    Tries to load the in-cluster configuration, and if it fails, tries to load the kubeconfig file.
    
    Returns:
        tuple: A tuple containing the loaded configuration object and a string indicating the type of configuration loaded.
    """
    logger.info("Attempting to load Kubernetes configuration...")
    try:
        config.load_incluster_config()
        k8s_config = client.Configuration()  # Get the configuration details
        config_type = "in-cluster"
        logger.info("Loaded in-cluster configuration.")
    except config.ConfigException as e:
        logger.debug("Could not load in-cluster configuration. Attempting to load kubeconfig file...")
        try:
            config.load_kube_config()
            k8s_config = client.Configuration()  # Get the configuration details
            config_type = "kubeconfig"
            logger.info("Loaded kubeconfig file.")
        except config.ConfigException as e:
            logger.error("Could not load Kubernetes configuration: %s", e)
            raise
    logger.debug("Kubernetes configuration loaded successfully.")
    return config_type, k8s_config


async def decode_k8s_jwt_payload(jwt_token: str) -> Union[dict, None]:
    """Decodes the payload from a K8s JWT token."""
    logger.info("Decoding JWT token payload...")
    try:
        # Split the JWT token to extract the payload
        payload = jwt_token.split('.')[1]
        logger.debug(f"Extracted payload: {payload}")

        # Calculate the required padding for Base64 decoding
        padding = '=' * (4 - len(payload) % 4)
        logger.debug(f"Calculated padding: {padding}")

        # Decode the Base64 payload and convert it to JSON
        payload_json = base64.b64decode(payload + padding).decode('utf-8')
        logger.debug(f"Decoded payload JSON: {payload_json}")

        return json.loads(payload_json)
    except (IndexError, json.JSONDecodeError, base64.binascii.Error) as e:
        logger.error(f"Error decoding JWT token: {e}")
        return None


async def _handle_error(error: ToolException) -> str:
    return (
        "The following errors occurred during tool execution:"
        + error.args[0]
        + "Please try another tool."
    )


async def fetch_service_account_info() -> str:
    """
    This function fetches the service account details from the Kubernetes environment.
    It reads the namespace and service account name from the respective files in the Kubernetes secrets directory.

    Returns:
        str: A JSON stringified dictionary containing the 'namespace' and 'service_account_name'.
    """
    logger.info("Fetching service account info...")

    # Path to the service account token and namespace file
    token_path: str = '/var/run/secrets/kubernetes.io/serviceaccount/token'

    # Initialize the result dictionary
    result: dict = {
        'namespace': None,  # The namespace in which the service account is defined
        'service_account_name': None  # The name of the service account
    }

    # Read and decode the JWT token
    try:
        with open(token_path, 'r') as file:
            jwt_token: str = file.read().strip()
            logger.debug("JWT token read successfully.")
            # Split the JWT token which is in 'header.payload.signature' format
            payload: str = jwt_token.split('.')[1]
            # Correct padding for Base64 decoding
            payload += '=' * (-len(payload) % 4)
            payload_json: str = base64.urlsafe_b64decode(payload).decode('utf-8')
            payload_data: dict = json.loads(payload_json)
            # Extract service account name from the token
            result['service_account_name'] = payload_data['kubernetes.io']["serviceaccount"]["name"]
            result['namespace'] = payload_data['kubernetes.io']["namespace"]
            logger.debug(f"Service account name extracted: {result['service_account_name']}")
    except IOError as e:
        logger.error(f"Could not read token from file: {e}")
    except (IndexError, json.JSONDecodeError, base64.binascii.Error) as e:
        logger.error(f"Error decoding JWT token: {e}")

    logger.info("Service account info fetched successfully.")
    return json.dumps(result)


async def generate_default_deploy_role_manifest(namespace: str, service_account_name: str) -> dict:
    """
    Generates a role manifest with default resources and verbs for deploying to a namespace.
    This is a convenience function that provides a default set of permissions typically needed for deployments.
    """
    resources = ["pods", "pods/log", "services", "deployments", "secrets"]
    verbs = ["get", "watch", "list", "create", "delete", "update", "patch"]
    return await generate_role_manifest(namespace, service_account_name, resources, verbs)


async def generate_role_manifest(namespace: str, service_account_name: str, resources: list, verbs: list) -> dict:
    logger.debug("Generating role manifest...")

    # Define the permissions required in the Role
    rules: list = [
        {
            "apiGroups": [""],  # "" indicates the core API group
            "resources": resources,
            "verbs": verbs
        },
        # Add more rules here as needed
    ]

    logger.info("Permissions defined for the Role.")
    # Create the Role manifest
    role_manifest: dict = {
        "apiVersion": "rbac.authorization.k8s.io/v1",
        "kind": "Role",
        "metadata": {
            "namespace": namespace,
            "name": f"{service_account_name}-role"
        },
        "rules": rules
    }

    logger.info("Role manifest created successfully.")
    return role_manifest

async def can_i_deploy_into_namespace(namespace: str) -> bool:
    resource_attributes = {
        "secrets": ["create", "get", "list", "watch", "update", "patch", "delete"],
        "services": ["create", "get", "list", "watch", "update", "patch", "delete"],
        "deployments": ["create", "get", "list", "watch", "update", "patch", "delete"]
    }
    return await can_i(namespace, resource_attributes)



async def can_i(namespace: str, resource_attributes: dict) -> bool:
    """
    This function checks if the current service account has the specified permissions in the given namespace.
    
    Args:
        namespace (str): The namespace to check permissions in.
        resource_attributes (dict): The resource attributes to check permissions for. 
            It should contain 'verb' and 'resource' keys, where the values are either a single string or a list of strings.
    
    Returns:
        bool: True if the service account has the specified permissions, False otherwise.
    """
    # Define the necessary variables
    token_file: str = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    ca_cert: str = '/var/run/secrets/kubernetes.io/serviceaccount/ca.crt'
    api_server: str = 'https://kubernetes.default.svc'
    
    # Read the ServiceAccount token
    try:
        with open(token_file, 'r') as file:
            token: str = file.read().replace('\n', '')
    except IOError as e:
        print(f"Could not read token from file: {e}")
        return False
    
    # Prepare the headers and the URL
    headers: dict = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
    }
    url: str = f'{api_server}/apis/authorization.k8s.io/v1/selfsubjectaccessreviews'
    
    # Prepare the data payload
    data: dict = {
        "apiVersion": "authorization.k8s.io/v1",
        "kind": "SelfSubjectAccessReview",
        "spec": {
            "resourceAttributes": {
                "namespace": namespace,
            }
        }
    }
    
    # Support multiple resourceAttributes if provided
    for key, value in resource_attributes.items():
        if isinstance(value, list):
            for val in value:
                data['spec']['resourceAttributes'][key] = val
                response: bool = await make_post_request(url, data, headers, ca_cert)
                if not response:
                    return False
        else:
            data['spec']['resourceAttributes'][key] = value
            response: bool = await make_post_request(url, data, headers, ca_cert)
            if not response:
                return False
                
    return True

# Example usage:
# Check if can create and delete pods and services
# print(can_i('default', {'verb': ['create', 'delete'], 'resource': ['pods', 'services']}))

async def make_post_request(url: str, data: dict, headers: dict, ca_cert: str) -> bool:
    """
    This function makes a POST request to the specified URL with the provided data, headers, and CA certificate.
    It then parses the response and returns a boolean value based on the status code and the 'allowed' status in the response.
    
    Args:
        url (str): The URL to make the POST request to.
        data (dict): The data to include in the POST request.
        headers (dict): The headers to include in the POST request.
        ca_cert (str): The path to the CA certificate for SSL verification.
    
    Returns:
        bool: True if the status code is 200 and 'allowed' is True in the response, False otherwise.
    """
    # Make the POST request
    response = requests.post(url, json=data, headers=headers, verify=ca_cert)
    
    # Add debug statement for response status code
    logger.debug(f'Debug: Received status code {response.status_code}')
    
    # Parse the response
    if response.status_code // 100 == 2 and response.json()['status']['allowed']:
        return True
    else:
        # Add info statement for error response
        logger.debug(f'Info: Error response content: {response.content}')
        return False


async def create_temp_helm_values(values_dict: dict, file_extension: str = '.yaml', debug: bool = False) -> str:
    """
    Create a temporary file for Helm chart values.

    :param values_dict: A dictionary containing the values to be written to the file.
    :param file_extension: The extension for the temporary file.
    :param debug: If True, print the path of the temporary file.
    :return: The path to the temporary file.
    """
    try:
        # Create a temporary file
        fd, temp_path = tempfile.mkstemp(suffix=file_extension, prefix='helm-values-')
    
        # Write the values dictionary to the file
        with os.fdopen(fd, 'w') as tmp_file:
            yaml.dump(values_dict, tmp_file, default_flow_style=False)
    
        if debug:
            print(f"Temporary file created at: {temp_path}")
        return temp_path
    except Exception as e:
        print(f"Error creating temporary file: {e}")
        raise


async def deploy_helm_chart_with_values(release_name: str, namespace: str, chart_url: str, chart_version: str, values_dict: dict, debug: bool = False) -> str:
    """
    Deploy a Helm chart into a Kubernetes cluster using Helm 3 with the specified values.
    """
    return await _deploy_helm_chart("gw",namespace,chart_url)


async def _deploy_helm_chart(release_name, namespace, chart_url, chart_version, values_file=None) -> str:
    """
    Deploy a Helm chart into a Kubernetes cluster using Helm 3.

    :param release_name: Name of the release.
    :param namespace: Kubernetes namespace where the chart will be installed.
    :param chart_url: URL to the chart repository or the package itself.
    :param chart_version: Version of the chart to deploy.
    :param values_file: (Optional) Path to a YAML file with values to override in the chart.
    """
    
    # Construct the Helm command
    cmd = [
        'helm', 'upgrade', release_name, chart_url,
        '--install',
        '--version', chart_version,
        '--namespace', namespace,
        '--wait'  # Wait for the installation to be complete
    ]
    
    # Add values file if provided
    if values_file:
        cmd += ['--values', values_file]
    
    # Deploy the chart
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Chart {release_name} installed successfully.\n{result.stdout}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to install chart {release_name}.\nError: {e.stderr}")
        raise



async def deploy_akeyless_gateway(target_namespace: str, admin_access_id: str, release_name: str = "gw", isGKE_workload_identity: bool = True) -> str:
    """
    Asynchronously deploys the Akeyless Gateway Helm chart into a specified Kubernetes namespace. This function fetches the latest version of the chart from a remote repository, installs or upgrades the release with custom values, and returns the deployment result in JSON format.

    Args:
        target_namespace (str): The Kubernetes namespace where the Helm chart will be deployed.
        admin_access_id (str): The Akeyless admin access ID used for authentication within the Akeyless Gateway.
        release_name (str, optional): The release name for the Helm chart. Defaults to "gw".
        isGKE_workload_identity (bool, optional): A flag indicating whether to configure the deployment for GKE workload identity. Defaults to True.

    Returns:
        str: A JSON string containing details about the Helm chart deployment, including the release name, namespace, revision number, and deployment status.
    """
    # Initialize the Kubernetes client
    client = Client()
    
    # Fetch the latest version of the Akeyless Helm chart
    latest_akeyless_helm_chart_version = await get_latest_akeyless_helm_chart_release_number()

    # Retrieve the Helm chart
    chart = await client.get_chart(
        "akeyless-api-gateway",
        repo="https://akeylesslabs.github.io/helm-charts",
        version=latest_akeyless_helm_chart_version
    )
    
    # Prepare custom values for the Helm chart deployment
    values = {
                "akeylessUserAuth": {"adminAccessId": admin_access_id},
                "livenessProbe": {"initialDelaySeconds": 30},
                "readinessProbe": {"initialDelaySeconds": 30}
            }

    # Configure node selector for GKE workload identity if enabled
    if isGKE_workload_identity:
        values["deployment"] = {
            "nodeSelector": {"iam.gke.io/gke-metadata-server-enabled": "true"},
            "service_account": {
                "serviceAccountName": "heimdal-sa"
            }
        }

    # Install or upgrade the Helm chart release
    revision = await client.install_or_upgrade_release(
        release_name,
        chart,
        target_namespace,
        values,
        atomic=True,
        wait=False
    )

    # Log the details of the deployed release
    print(
        f"Release name: {revision.release.name}",
        f"Namespace: {revision.release.namespace}",
        f"Revision: {revision.revision}",
        f"Status: {str(revision.status)}"
    )

    # Compile the deployment result into a dictionary
    results = {
        "revision_release_name": revision.release.name,
        "revision_namespace": revision.release.namespace,
        "revision": revision.revision,
        "status": str(revision.status)
    }

    # Return the deployment result as a JSON string
    return json.dumps(results)



async def get_deployed_helm_releases(namespace: str) -> List[str]:
    """
    This function uses the pyhelm3 library to find out what Helm releases have been deployed into a namespace.
    It returns a list of release names that are currently deployed in the given namespace.

    Args:
        namespace (str): The Kubernetes namespace to check for deployed Helm releases.

    Returns:
        List[str]: A list of Helm release names deployed in the namespace.
    """
    # Initialize the Helm client
    client = Client()

    # List all releases in the namespace
    releases = client.list_releases(namespace=namespace)

    # Extract the release names from the releases
    release_names = [release.name for release in releases]

    return release_names


async def get_latest_akeyless_helm_chart_release_number() -> str:
    """
    This function returns the latest release number of the Akeyless Helm chart from the Akeyless Helm repository.
    """
    return await get_latest_helm_chart_release_number("akeyless-api-gateway", "https://akeylesslabs.github.io/helm-charts")


async def get_latest_helm_chart_release_number(chart_ref:str, chart_repo:str) -> str:
    """
    This function returns the latest release number of a Helm chart from a Helm repository.
    It attempts to handle exceptions and errors more gracefully to be less brittle.
    """
    # Initialize the Helm client
    client = Client()

    try:
        # Fetch the Helm chart
        chart = await client.get_chart(
            chart_ref,
            repo=chart_repo
        )
        # Return the latest release number if available
        if chart and chart.metadata and chart.metadata.version:
            return chart.metadata.version
        else:
            raise ValueError("Chart metadata or version is missing.")
    except Exception as e:
        # Log the error for debugging purposes
        logging.error(f"Failed to fetch the latest Helm chart release number for {chart_ref} from {chart_repo}: {e}")
        # Return a default value or raise an exception as per your error handling policy
        return "unknown"

import requests
import concurrent.futures
from typing import Optional


def check_aws() -> Optional[str]:
    """
    This function checks if the script is running on AWS by trying to reach the AWS metadata endpoint.
    It first requests a token and then uses this token to request the metadata.
    If the metadata request is successful, it returns 'aws'.
    If any request fails, it returns None.
    
    Returns:
        Optional[str]: 'aws' if the script is running on AWS, None otherwise.
    """
    try:
        # Request a token from the AWS metadata endpoint
        token_response: requests.Response = requests.put(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            timeout=2
        )
        # If the token request is successful, use the token to request the metadata
        if token_response.status_code == 200:
            aws_response: requests.Response = requests.get(
                "http://169.254.169.254/latest/meta-data/",
                headers={"X-aws-ec2-metadata-token": token_response.text},
                timeout=2
            )
            # If the metadata request is successful, return 'aws'
            if aws_response.status_code == 200:
                return 'aws'
    except requests.exceptions.RequestException:
        # If any request fails, return None
        pass
    return None

def check_azure() -> Optional[str]:
    """
    This function checks if the script is running on Azure by trying to reach the Azure metadata endpoint.
    If the metadata request is successful, it returns 'azure'.
    If the request fails, it returns None.
    
    Returns:
        Optional[str]: 'azure' if the script is running on Azure, None otherwise.
    """
    try:
        # Prepare the request to the Azure metadata endpoint
        url: str = "http://169.254.169.254/metadata?api-version=2021-02-01&format=text"
        headers: dict = {"Metadata": "true"}
        proxies: dict = {"http": None, "https": None}
        timeout: int = 2
        
        # Make the request
        azure_response: requests.Response = requests.get(
            url,
            headers=headers,
            proxies=proxies,
            timeout=timeout
        )
        
        # If the metadata request is successful, return 'azure'
        if azure_response.status_code == 200:
            return 'azure'
    except requests.exceptions.RequestException:
        # If the request fails, return None
        pass
    return None

def check_gcp() -> Optional[str]:
    """
    This function checks if the script is running on GCP by trying to reach the GCP metadata endpoint.
    If the metadata request is successful, it returns 'gcp'.
    If the request fails, it returns None.
    
    Returns:
        Optional[str]: 'gcp' if the script is running on GCP, None otherwise.
    """
    try:
        # Prepare the request to the GCP metadata endpoint
        url: str = "http://metadata.google.internal/computeMetadata/"
        headers: dict = {"Metadata-Flavor": "Google"}
        timeout: int = 2
        
        # Make the request
        gcp_response: requests.Response = requests.get(
            url,
            headers=headers,
            timeout=timeout
        )
        
        # If the metadata request is successful, return 'gcp'
        if gcp_response.status_code == 200:
            return 'gcp'
    except requests.exceptions.RequestException:
        # If the request fails, return None
        pass
    return None

def detect_cloud_provider() -> str:
    """
    Detects the cloud service provider where the script is running by concurrently 
    checking metadata endpoints of AWS, Azure, and GCP. It returns the provider 
    name if the respective metadata endpoint is reachable and responds successfully.
    If unknown is returned and the user is sure they are in a supported cloud provider
    they may have to enable cloud identity for that workload with their CSP.

    Returns:
        str: The cloud provider name ('aws', 'azure', 'gcp') or 'unknown' if detection fails.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Prepare the futures for each cloud provider check
        future_to_provider = {
            executor.submit(check_aws): 'aws',
            executor.submit(check_azure): 'azure',
            executor.submit(check_gcp): 'gcp',
        }
        
        # Iterate over the completed futures
        for future in concurrent.futures.as_completed(future_to_provider):
            provider: Optional[str] = future.result()
            # If a provider was detected, return it
            if provider:
                return provider
    # If no provider was detected, return 'unknown'
    return 'unknown'

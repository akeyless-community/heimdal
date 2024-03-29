from akeyless_cloud_id import CloudId
import logging
import asyncio
from heimdal.tools.utility_tools.akeyless_api_operations import extract_gcp_project_id
from heimdal.tools.utility_tools.kubernetes_operations import get_latest_akeyless_helm_chart_release_number

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the CloudId generator
cloud_id_generator = CloudId()

async def main():
    # Generate the cloud ID for GCP
    cloud_id: str = cloud_id_generator.generateGcp()
    # Extract the GCP project ID from the cloud ID
    gcp_project_id: str = await extract_gcp_project_id(cloud_id)
    logging.debug(f"GCP Project ID: {gcp_project_id}")

if __name__ == "__main__":
    asyncio.run(main())

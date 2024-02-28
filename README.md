# Heimdal - Agentic AI-Powered Helm Chart Installer

Welcome to Heimdal, the cutting-edge AI-powered assistant designed to streamline the deployment of the Akeyless Gateway into Kubernetes clusters. Heimdal leverages advanced language models and a suite of tools to provide an intuitive and automated installation experience.

## Features

- **Environment Scanning**: Automatically detects the Kubernetes namespace and service account of the deployment.
- **Cloud Service Provider Detection**: Identifies the cloud service provider and creates the appropriate Akeyless authentication method.
- **Helm Chart Deployment**: Deploys the Akeyless Gateway Helm chart using the generated authentication method Access ID.
- **Comprehensive Reporting**: Returns detailed information about the deployment, including namespace, service account, and cloud provider.

## Getting Started

To use Heimdal, you will need:

- An Akeyless Token from the Akeyless web console.
- Permissions for Heimdal to deploy the Helm chart into the Kubernetes namespace.

## Usage

1. **Set Akeyless Token**: Enter your Akeyless Token in the Streamlit app sidebar.
2. **Approve Installation**: Click the button in the Streamlit app to start the installation process.
3. **Monitor Progress**: Follow the on-screen instructions and monitor the deployment process.
4. **Review Details**: After deployment, review the details of the installation provided by Heimdal.

## Development

Heimdal is built using Python and integrates with Langchain and OpenAI's language models. The project structure includes:

- `app.py`: The main application script with the AI logic and tool integrations.
- `streamlit_app.py`: The Streamlit interface for user interaction.
- `requirements.txt`: The list of Python package dependencies.

To contribute to the project, clone the repository, create a virtual environment, install dependencies, and start adding your features or improvements.

## Support

For support, please open an issue in the project's GitHub repository or reach out to the maintainers.

## License

Heimdal is open-source software licensed under the Apache License 2.0.

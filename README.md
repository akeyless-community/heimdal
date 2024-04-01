# Heimdal: AI-Powered Kubernetes Helm Chart Deployment

Heimdal is an innovative open-source project that leverages the power of Large Language Models (LLMs) and the LangChain framework to automate Kubernetes Helm chart deployments. By integrating cutting-edge AI technologies with DevOps practices, Heimdal aims to demonstrate a path to revolutionize the way applications are deployed and managed in cloud-native environments.

## Features

- AI-driven conversational agent for intuitive deployment commands
- Intelligent interpretation and execution of user intents
- Seamless integration with Kubernetes and Helm
- Automated verification of permissions and cloud service provider detection
- Secure management of Kubernetes secrets
- User-friendly interface powered by Chainlit and LangGraph
- Modular architecture for easy customization and extensibility

## Getting Started

### Prerequisites

- Kubernetes cluster
- Helm package manager
- Python 3.7+
- Required environment variables (see `.env.example`)

### Installation

1. Clone the repository:
```sh
git clone https://github.com/akeyless-community/heimdal.git
```

2. Install the required dependencies:
```sh
pip install -r requirements.txt
```

3. Create a `.env` file based on the provided `.env.example` and set the required environment variables.

4. Deploy Heimdal using Kustomize:

```sh
kubectl apply -k .
```

5. Access the Heimdal UI by navigating to the provided URL.

## Usage

Heimdal provides a conversational interface for deploying the Akeyless Gateway Helm chart on Kubernetes. Simply interact with the AI agent through the Chainlit-powered UI, providing natural language commands and responding to prompts as needed. Heimdal will intelligently interpret your intents, verify permissions, detect the cloud service provider, and execute the necessary deployment steps.

## Contributing

We welcome contributions from the community! If you'd like to contribute to Heimdal, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Submit a pull request to the main repository.

Please ensure that your code adheres to the project's coding conventions and includes appropriate tests.

## License

Heimdal is released under the [Apache License 2.0](LICENSE).

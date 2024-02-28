from heimdal.tools.utility_tools.kubernetes_operations import can_i_deploy_into_namespace, deploy_akeyless_gateway, fetch_service_account_info, generate_k8s_secret_from_literal_values, get_deployed_helm_releases
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
import json
from langchain_core.messages import BaseMessage
from pyhelm3 import Client
from typing import Callable, List, TypedDict, Sequence
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.prebuilt import ToolExecutor

from langchain.tools import StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv
import os
import logging
import os
import json
import asyncio
from heimdal.tools.llm_tools.cloud_detection import detect_cloud_provider
from heimdal.tools.utility_tools.akeyless_api_operations import create_akeyless_api_key_auth_method, create_aws_cloud_auth_method, create_azure_cloud_auth_method, create_gcp_cloud_auth_method
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load .env file
load_dotenv(verbose=True)

# Get OpenAI keys from .env file
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_ORGANIZATION"] = os.getenv("OPENAI_ORGANIZATION")

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "Demos"

# Initialize OpenWeatherMap
# os.environ["AKEYLESS_TOKEN"] = os.getenv("AKEYLESS_TOKEN")

# Define the tools


# Get Pod Namespace and Service Account Tool
def get_pod_namespace_and_service_account_extractor() -> str:
    """
    This tool is used to extract namespace and service account information from Kubernetes.
    It makes an external API call to fetch the information.
    """
    logger.debug("Entering get_namespace_and_service_account_extractor")
    try:
        logger.info("Extracting Kubernetes information")
        k8s_info = fetch_service_account_info()
        logger.debug(f"Kubernetes info: {k8s_info}")
        k8s_info_dict = json.loads(k8s_info)
        logger.debug(f"Kubernetes info dictionary: {k8s_info_dict}")
        logger.info("Kubernetes information extraction complete")
        return json.dumps(k8s_info_dict)
    except Exception as e:
        error_message = {"error": str(e)}
        return json.dumps(error_message)


get_pod_namespace_and_service_account = StructuredTool.from_function(
    func=get_pod_namespace_and_service_account_extractor,
    name="Get_Pod_Namespace_And_Service_Account",
    description="Get the namespace and service account of the running pod",
    return_direct=False,
)


# Cloud Service Detector Tool
def cloud_service_detector_tool() -> str:
    """
    This tool is used to detect the cloud service provider.
    It makes an external API call to detect the provider.
    """
    logger.debug("Entering cloud_service_detector_tool")
    try:
        # Detect the cloud service provider
        cloud_service_provider: str = detect_cloud_provider()
        logger.debug(
            f"Detected cloud service provider: {cloud_service_provider}")
        return json.dumps({"cloud_service_provider": cloud_service_provider})
    except Exception as e:
        error_message = {"error": str(e)}
        logger.error("Error detecting cloud service provider: " + str(e))
        return json.dumps({"error": str(e)})


cloud_service_detector = StructuredTool.from_function(
    func=cloud_service_detector_tool,
    name="Cloud_Service_Detector",
    description="Detect the cloud service provider. Required to run BEFORE creating an authentication method.",
    return_direct=False,
)


def create_auth_method(auth_type: str, create_method: Callable) -> str:
    """
    Utility function to create an authentication method in Akeyless.
    It handles logging, error handling, and response formatting.

    :param auth_type: Type of the authentication method ('AWS' or 'Azure').
    :param create_method: Function to call for creating the authentication method.
    :param request: Request string.
    :return: JSON string with the result or error message.
    """
    logger.debug(f"Entering create_auth_method for {auth_type}")
    json_result_str: str = None
    try:
        # Call the provided method to create authentication method
        result = create_method()
        if result.startswith("p-"):
            logger.debug(f"{auth_type} authentication method created.")
            json_result_str = json.dumps(
                {"message": f"{auth_type} authentication method created.", "akeyless_access_id": result})
        else:
            logger.error(
                f"Failed to create {auth_type} authentication method.")
            json_result_str = json.dumps(
                {"error": f"Failed to create {auth_type} authentication method."})
    except Exception as e:
        error_message = {"error": str(e)}
        logger.error(
            f"Error creating {auth_type} authentication method: " + str(e))
        json_result_str = json.dumps(error_message)
    return json_result_str


# Akeyless API Key Auth Method Creator Tool
def aws_auth_method_creator_tool() -> str:
    """
    This tool is used to create an AWS authentication method in Akeyless.
    It makes an external API call to create the method.
    """
    return create_auth_method('AWS', create_aws_cloud_auth_method)


aws_auth_method_creator = StructuredTool.from_function(
    func=aws_auth_method_creator_tool,
    name="AWS_Auth_Method_Creator",
    description="Create an AWS authentication method in Akeyless. Only create this if the cloud service provider is known to be AWS. If the cloud service provider is not AWS then DO NOT create this authentication method.",
    return_direct=False,
)


# Akeyless Azure Auth Method Creator Tool
def azure_auth_method_creator_tool() -> str:
    """
    This tool is used to create an Azure authentication method in Akeyless.
    It makes an external API call to create the method.
    """
    return create_auth_method('Azure', create_azure_cloud_auth_method)


azure_auth_method_creator = StructuredTool.from_function(
    func=azure_auth_method_creator_tool,
    name="Azure_Auth_Method_Creator",
    description="Create an Azure authentication method in Akeyless. Only create this if the cloud service provider is known to be Azure. If the cloud service provider is not Azure then DO NOT create this authentication method.",
    return_direct=False,
)


# Akeyless GCP Auth Method Creator Tool
def gcp_auth_method_creator_tool() -> str:
    """
    This tool is used to create an GCP authentication method in Akeyless.
    It makes an external API call to create the method.
    """
    return create_auth_method('GCP', create_gcp_cloud_auth_method)


gcp_auth_method_creator = StructuredTool.from_function(
    func=gcp_auth_method_creator_tool,
    name="GCP_Auth_Method_Creator",
    description="Create an GCP authentication method in Akeyless. Only create this if the cloud service provider is known to be GCP. If the cloud service provider is not GCP then DO NOT create this authentication method.",
    return_direct=False,
)


# Akeyless API Key Auth Method Creator Tool
def api_key_auth_method_creator_tool() -> str:
    """
    This tool is used to create an Azure authentication method in Akeyless.
    It makes an external API call to create the method.
    """
    return create_auth_method('API Key', create_akeyless_api_key_auth_method)


api_key_auth_method_creator = StructuredTool.from_function(
    func=api_key_auth_method_creator_tool,
    name="API_Key_Auth_Method_Creator",
    description="Create an API Key authentication method in Akeyless. Only create this if the cloud service provider is unknown. If the cloud service provider is known, DO NOT create this authentication method.",
    return_direct=False,
)


# Kubernetes Secret Deployer Tool
class KubernetesSecret(BaseModel):
    secret_name: str = Field(description="Name of the secret")
    namespace: str = Field(
        description="Namespace in which the secret will be deployed")
    literal_values: dict = Field(
        description="Literal values to be used in the secret")


def kubernetes_secret_deployer_tool(secret_name: str, namespace: str, literal_values: dict) -> str:
    """
    This tool is used to deploy a Kubernetes secret.
    It makes an external API call to create the secret.
    """
    try:

        logger.debug("Entering kubernetes_secret_deployer_tool")
        result = generate_k8s_secret_from_literal_values(
            secret_name, namespace, literal_values)
        return json.dumps(result)
    except Exception as e:
        error_message = {"error": str(e)}
        logger.error("Error deploying Kubernetes secret: " + str(e))
        return json.dumps(error_message)


kubernetes_secret_deployer = StructuredTool.from_function(
    func=kubernetes_secret_deployer_tool,
    name="Kubernetes_Secret_Deployer",
    description="Deploy a Kubernetes secret. Only create this if the cloud service provider is unknown. If the cloud service provider is known, DO NOT create kuberenetes secret.",
    args_schema=KubernetesSecret,
    return_direct=False,
)


class HelmChartDeployment(BaseModel):
    namespace: str = Field(
        description="Namespace in which the chart will be deployed")
    auth_method_id: str = Field(
        description="Akeyless access id for auth method")
    release_name: str = Field(
        default="gw",
        description="Optional: The release name for the Helm chart. Defaults to 'gw' if not provided.")

def helm_chart_deployer_tool(namespace: str, auth_method_id: str, release_name: str = "gw") -> str:
    """
    This tool is used to deploy the Akeyless Gateway Helm chart in a Kubernetes cluster.
    It makes an external API call to the k8s cluster to deploy the chart.
    The release name is optional and defaults to "gw".

    :param namespace: The namespace in which the chart will be deployed.
    :param auth_method_id: The Akeyless admin access ID to be used for authentication.
    :param release_name: Optional. The release name for the Helm chart. If not specified, "gw" will be used as the default release name.
    :return: A JSON string with the result of the deployment.
    """
    try:
        logger.debug(f"Deploying Helm chart with release name '{release_name}', auth method id: '{auth_method_id}' in namespace: '{namespace}'")
        result = asyncio.run(
            deploy_akeyless_gateway(namespace, auth_method_id, release_name))
        return result
    except Exception as e:
        error_message = {"error": str(e)}
        logger.error(f"Error deploying Helm chart: {e}")
        return json.dumps(error_message)


helm_chart_deployer = StructuredTool.from_function(
    func=helm_chart_deployer_tool,
    name="Helm_Chart_Deployer",
    description="Deploy the Akeyless Gateway Helm chart in a Kubernetes cluster",
    args_schema=HelmChartDeployment,
    return_direct=False,
)


def can_i_deploy_into_namespace_tool(namespace: str) -> str:
    """
    This tool is used to check if the bot can deploy into the namespace.
    It makes an external API call to check if the bot has the necessary permissions.
    """
    hasDeploymentPermission = can_i_deploy_into_namespace(namespace)
    return json.dumps({"can_i_deploy": hasDeploymentPermission})


can_i_deploy_into_namespace_checker = StructuredTool.from_function(
    func=can_i_deploy_into_namespace_tool,
    name="Can_I_Deploy_Into_Namespace",
    description="Check if the bot has permission to deploy the helm chart into the namespace. You can only run this tool AFTER you have detected the namespace and service account. If the bot does not have permission to deploy into the namespace, then do not deploy the helm chart.",
    return_direct=False,
)


def get_list_of_helm_releases_in_namespace_tool(namespace: str) -> str:
    """
    This tool is used to get the list of helm releases in a namespace.
    It makes an external API call to get the list of releases.
    """
    helm_releases_in_namespace: List[str] = get_deployed_helm_releases(namespace)
    return json.dumps({"helm_releases_in_namespace": helm_releases_in_namespace})


get_list_of_helm_releases_in_namespace = StructuredTool.from_function(
    func=get_list_of_helm_releases_in_namespace_tool,
    name="Get_List_Of_Helm_Releases_In_Namespace",
    description="Get the list of helm releases in a namespace. This tool can be used to determine if the anticipated helm chart release name is already taken and a new name needs to be generated and used.",
    return_direct=False,
)   

tools = [
    get_pod_namespace_and_service_account,
    can_i_deploy_into_namespace_checker,
    cloud_service_detector,
    aws_auth_method_creator,
    azure_auth_method_creator,
    gcp_auth_method_creator,
    # api_key_auth_method_creator,
    # kubernetes_secret_deployer,
    helm_chart_deployer,
    get_list_of_helm_releases_in_namespace
]

tool_executor = ToolExecutor(tools)


class Response(BaseModel):
    """Final answer to the user"""
    namespace: str = Field(
        description="Namespace in which the chart was deployed")
    service_account: str = Field(
        description="Service account used to deploy the chart")
    detected_cloud_service_provider: str = Field(
        description="Detected cloud service provider")
    can_i_deploy: str = Field(
        description="Can the bot deploy into the namespace?")
    akeyless_access_id: str = Field(
        description="Akeyless access id for auth method")
    release_name: str = Field(description="Name of the release")
    revision_namespace: str = Field(description="Namespace of the release")
    revision: int = Field(description="Revision number")
    status: str = Field(description="Status of the release")


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Heimdal, a prototype AI-powered assistant designed to help deploy the Akeyless Gateway into a kubernetes cluster. You have access to a variety of tools to help you accomplish this task.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

# Create the OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

# demo_ephemeral_chat_history_for_chain = ChatMessageHistory()

# Create the tools to bind to the model
tools = [convert_to_openai_function(t) for t in tools]
tools.append(convert_to_openai_function(Response))

# MODFIICATION: we're using bind_tools instead of bind_function
model = {"messages": RunnablePassthrough()} | prompt | llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


# Define the function that determines whether to continue or not
def should_continue(state):
    last_message = state["messages"][-1]
    # If there are no tool calls, then we finish
    if "tool_calls" not in last_message.additional_kwargs:
        return "end"
    # If there is a Response tool call, then we finish
    elif any(
        tool_call["function"]["name"] == "Response"
        for tool_call in last_message.additional_kwargs["tool_calls"]
    ):
        return "end"
    # Otherwise, we continue
    else:
        return "continue"


# Define the function that calls the model
def call_model(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": messages + [response]}


# Define the function to execute tools
def call_tool(state):
    messages = state["messages"]
    # We know the last message involves at least one tool call
    last_message = messages[-1]

    # We loop through all tool calls and append the message to our message log
    for tool_call in last_message.additional_kwargs["tool_calls"]:
        action = ToolInvocation(
            tool=tool_call["function"]["name"],
            tool_input=json.loads(tool_call["function"]["arguments"]),
            id=tool_call["id"],
        )

        # We call the tool_executor and get back a response
        response = tool_executor.invoke(action)
        # We use the response to create a FunctionMessage
        function_message = ToolMessage(
            content=str(response), name=action.tool, tool_call_id=tool_call["id"]
        )

        # Add the function message to the list
        messages.append(function_message)

    # We return a list, because this will get added to the existing list

    return {"messages": messages}


# Initialize a new graph
graph = StateGraph(AgentState)

# Define the two Nodes we will cycle between
graph.add_node("agent", call_model)
graph.add_node("action", call_tool)

# Set the Starting Edge
graph.set_entry_point("agent")

# Set our Contitional Edges
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# Set the Normal Edges
graph.add_edge("action", "agent")

# memory = SqliteSaver.from_conn_string(":memory:")

# Compile the workflow
app = graph.compile()


class AppInteractor:
    def __init__(self, app, llm, tools):
        self.app = app
        self.llm = llm
        self.tools = tools


app_interactor = AppInteractor(app, llm, tools)

# inputs = {
#     "messages": [
#         HumanMessage(
#             content="""Please check what the kubernetes namespace and service account are for this bot to deploy the akeyless gateway helm chart. Please make sure the bot has permission to deploy the helm chart into this namespace. Please figure out the cloud service provider where this bot is installed and create the appropriate authentication method. Please deploy the helm chart using the authentication method access ID. Once the helm chart is deployed, please return all the details about everything created."""
#         )
#     ]
# }
# for output in app.with_config({"run_name": "Heimdal with Tools", "recursion_limit": 50}).stream(inputs):
#     # stream() yields dictionaries with output keyed by node name
#     for key, value in output.items():
#         print(f"Output from node '{key}':")
#         print("---")
#         print(value)
#     print("\n---\n")

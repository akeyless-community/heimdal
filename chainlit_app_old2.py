import asyncio
import datetime
from langchain_core.tools import tool
from typing import Annotated, Callable
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import BaseMessage
from langchain.chains.llm_math.base import LLMMathChain
from langchain.tools import StructuredTool
import operator
import chainlit as cl
from chainlit.input_widget import TextInput
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.agents import create_json_chat_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.agents import create_json_chat_agent
from akeyless.models import ValidateTokenOutput
import json
from langchain_core.tools import BaseTool
import os
from typing import List, Union
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ChatMessage,
    SystemMessage,
    FunctionMessage,
    ToolMessage,
)
import logging
from dotenv import load_dotenv
from langchain.pydantic_v1 import BaseModel, Field
from heimdal.tools.llm_tools.cloud_detection import detect_cloud_provider

from heimdal.tools.utility_tools.akeyless_api_operations import check_if_akeyless_auth_method_exists_from_list_auth_methods, create_akeyless_api_key_auth_method, create_aws_cloud_auth_method, create_azure_cloud_auth_method, create_gcp_cloud_auth_method, validate_akeyless_token
from heimdal.tools.utility_tools.kubernetes_operations import can_i_deploy_into_namespace, deploy_akeyless_gateway, fetch_service_account_info, generate_k8s_secret_from_literal_values, get_deployed_helm_releases
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

DEPLOY_GATEWAY_COMMAND = "Please check what the kubernetes namespace and service account are for this bot to deploy the akeyless gateway helm chart. Please make sure the bot has permission to deploy the helm chart into this namespace. Please figure out the cloud service provider where this bot is installed and create the appropriate authentication method. Please deploy the helm chart using the authentication method access ID. Once the helm chart is deployed, please return all the details about everything created."

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

class HumanInputChainlit(BaseTool):
    """Tool that adds the capability to ask user for input."""

    name = "human"
    description = (
        "You can ask a human for guidance or information when you think you "
        "got stuck or you are not sure what to do next. "
        "The input should be a question for the human."
    )

    def _run(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        try:
            reply = cl.run_sync(cl.AskUserMessage(content=query, timeout=120, raise_on_timeout=True).send())
            return reply["output"].strip()
        except Exception as e:
            logging.error(f"Failed to get human input: {str(e)}")
            return "Error: Failed to get human input, maybe try asking again letting the user know of the timeout of 2 minutes."

    async def _arun(
        self,
        query: str,
        run_manager=None,
    ) -> str:
        """Use the Human input tool."""
        try:
            res = await cl.AskUserMessage(content=query).send()
            return res["output"]
        except Exception as e:
            logging.error(f"Failed to get human input asynchronously: {str(e)}")
            return "Error: Failed to get human input."


@cl.action_callback("Approve the Scanning of the environment and install the Akeyless Gateway")
async def on_action(action):
    # Add the DEPLOY_GATEWAY_COMMAND to the message history as if the user sent it
    message = cl.Message(
        content=DEPLOY_GATEWAY_COMMAND,
    )
    await message.send()
    await main(message)
    # Remove the action button from the chatbot user interface
    await action.remove()


@cl.on_settings_update
async def setup_agent(settings):
    """
    Handles the event when chat settings are updated. Specifically, it checks for the 'akeyless_token' in the settings,
    validates it, and sets it as an environment variable if valid.

    Args:
        settings (dict): A dictionary containing the updated settings.
    """
    # Debug statement to log the settings update event and its content
    cl.logger.debug(f"on_settings_update triggered with settings: {settings}")
    
    # Extract the 'akeyless_token' from the settings
    akeyless_token = settings.get("akeyless_token", os.getenv("AKEYLESS_TOKEN", ""))
    
    # Prepare the error message in case the token does not meet the requirements
    token_error_msg = cl.Message(content="The Akeyless Token must be set within chat settings and it must start with 't-'")
    
    # Validate the 'akeyless_token' to ensure it starts with 't-'
    if not akeyless_token.startswith("t-"):
        # If the token is invalid, send the error message to the user
        await token_error_msg.send()
        # Log the event of an invalid token being provided
        cl.logger.info("Provided Akeyless Token is invalid. It must start with 't-'.")
    else:
        os.environ["AKEYLESS_TOKEN"] = ""
        validation_result: ValidateTokenOutput;
        
        try:
            validation_result = await validate_akeyless_token(akeyless_token)
        except Exception as e:
            logging.error(f"Failed to validate Akeyless token: {str(e)}")
            return "Error: Failed to validate Akeyless token."
        
        if validation_result.is_valid:
            # If the token is valid, set it as an environment variable
            os.environ["AKEYLESS_TOKEN"] = akeyless_token
            # the date looks like "2024-03-29 06:34:40 +0000 UTC"
            expiration_date_str = validation_result.expiration
            expiration_date = datetime.datetime.strptime(expiration_date_str, "%Y-%m-%d %H:%M:%S %z %Z")
            current_date = datetime.datetime.now(datetime.timezone.utc)
            time_until_expiration = expiration_date - current_date
            hours, remainder = divmod(time_until_expiration.seconds, 3600)
            minutes, _ = divmod(remainder, 60)
            # Handle calculating the expiration in UTC
            expiration_date_str = expiration_date.strftime("%Y-%m-%d %H:%M:%S %Z")
            
            # Create an action button to scan the environment and deploy the Akeyless Gateway
            actions = [
                cl.Action(
                    name="Approve the Scanning of the environment and install the Akeyless Gateway",
                    description="Approve the Scanning of the environment and install the Akeyless Gateway", 
                    value=akeyless_token)
            ]
            
            await cl.Message(content=f"A valid Akeyless Token was set successfully!\n\nThe token will expire in {hours} hours and {minutes} minutes or at {expiration_date_str}.\n\nWould you like to approve the scanning of the environment and install the Akeyless Gateway?\n", actions=actions).send()
        else:
            await cl.Message(content="The provided Akeyless Token is not valid. Please try again.").send()
        
        # Log the successful setting of the Akeyless Token
        cl.logger.info("Akeyless Token set successfully as an environment variable.")


@cl.on_chat_start
async def start():
    await cl.ChatSettings(
        [
            TextInput(
                id="akeyless_token",
                label="Akeyless Token",
                type="textinput",
                placeholder="t-fds023fsfs33...",
                description="You can retrieve the token from the Akeyless web console by clicking the top right hand corner down arrow and then choosing 'Copy token'")
        ]
    ).send()
    
    # Create a new instance of the chat memory
    chat_history = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("chat_history", chat_history)

    repl = PythonREPL()
    
    

    chat_model = ChatOpenAI(
            temperature=0,
            streaming=True,
            model="gpt-4-turbo-preview"
        )

    llm_math_chain = LLMMathChain.from_llm(llm=chat_model, verbose=True)
    
    # Tools Go here
    
    # Get Pod Namespace and Service Account Tool
    async def get_pod_namespace_and_service_account_extractor() -> str:
        """
        This tool is used to extract namespace and service account information from Kubernetes.
        It makes an external API call to fetch the information.
        """
        logger.debug("Entering get_namespace_and_service_account_extractor")
        try:
            logger.info("Extracting Kubernetes information")
            k8s_info = await fetch_service_account_info()
            logger.debug(f"Kubernetes info: {k8s_info}")
            k8s_info_dict = json.loads(k8s_info)
            logger.debug(f"Kubernetes info dictionary: {k8s_info_dict}")
            logger.info("Kubernetes information extraction complete")
            return json.dumps(k8s_info_dict)
        except Exception as e:
            error_message = {"error": str(e)}
            return json.dumps(error_message)


    get_pod_namespace_and_service_account = StructuredTool.from_function(
        coroutine=get_pod_namespace_and_service_account_extractor,
        name="Get_Pod_Namespace_And_Service_Account",
        description="Get the namespace and service account of the running pod",
        return_direct=False,
    )


    # Cloud Service Detector Tool
    async def cloud_service_detector_tool() -> str:
        """
        This tool is used to detect the cloud service provider.
        It makes an external API call to detect the provider.
        """
        logger.debug("Entering cloud_service_detector_tool")
        try:
            # Detect the cloud service provider
            cloud_service_provider: str = await detect_cloud_provider()
            logger.debug(
                f"Detected cloud service provider: {cloud_service_provider}")
            return json.dumps({"cloud_service_provider": cloud_service_provider})
        except Exception as e:
            error_message = {"error": str(e)}
            logger.error("Error detecting cloud service provider: " + str(e))
            return json.dumps({"error": str(e)})


    cloud_service_detector = StructuredTool.from_function(
        coroutine=cloud_service_detector_tool,
        name="Cloud_Service_Detector",
        description="Detect the cloud service provider. Required to run BEFORE creating an authentication method.",
        return_direct=False,
    )


    async def create_auth_method(auth_type: str, create_method: Callable) -> str:
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
            result = await create_method()
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
    async def aws_auth_method_creator_tool() -> str:
        """
        This tool is used to create an AWS authentication method in Akeyless.
        It makes an external API call to create the method.
        """
        return create_auth_method('AWS', create_aws_cloud_auth_method)


    aws_auth_method_creator = StructuredTool.from_function(
        coroutine=aws_auth_method_creator_tool,
        name="AWS_Auth_Method_Creator",
        description="Create an AWS authentication method in Akeyless. Only create this if the cloud service provider is known to be AWS. If the cloud service provider is not AWS then DO NOT create this authentication method.",
        return_direct=False,
    )


    # Akeyless Azure Auth Method Creator Tool
    async def azure_auth_method_creator_tool() -> str:
        """
        This tool is used to create an Azure authentication method in Akeyless.
        It makes an external API call to create the method.
        """
        return create_auth_method('Azure', create_azure_cloud_auth_method)


    azure_auth_method_creator = StructuredTool.from_function(
        coroutine=azure_auth_method_creator_tool,
        name="Azure_Auth_Method_Creator",
        description="Create an Azure authentication method in Akeyless. Only create this if the cloud service provider is known to be Azure. If the cloud service provider is not Azure then DO NOT create this authentication method.",
        return_direct=False,
    )


    # Akeyless GCP Auth Method Creator Tool
    async def gcp_auth_method_creator_tool() -> str:
        """
        This tool is used to create an GCP authentication method in Akeyless.
        It makes an external API call to create the method.
        """
        return create_auth_method('GCP', create_gcp_cloud_auth_method)


    gcp_auth_method_creator = StructuredTool.from_function(
        coroutine=gcp_auth_method_creator_tool,
        name="GCP_Auth_Method_Creator",
        description="Create an GCP authentication method in Akeyless. Only create this if the cloud service provider is known to be GCP. If the cloud service provider is not GCP then DO NOT create this authentication method. If this tool returns a conflict error about the name of the authentication method, then choose a different name and try again.",
        return_direct=False,
    )


    # Akeyless API Key Auth Method Creator Tool
    async def api_key_auth_method_creator_tool() -> str:
        """
        This tool is used to create an Azure authentication method in Akeyless.
        It makes an external API call to create the method.
        """
        return create_auth_method('API Key', create_akeyless_api_key_auth_method)


    api_key_auth_method_creator = StructuredTool.from_function(
        coroutine=api_key_auth_method_creator_tool,
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


    async def kubernetes_secret_deployer_tool(secret_name: str, namespace: str, literal_values: dict) -> str:
        """
        This tool is used to deploy a Kubernetes secret.
        It makes an external API call to create the secret.
        """
        try:

            logger.debug("Entering kubernetes_secret_deployer_tool")
            result = await generate_k8s_secret_from_literal_values(
                secret_name, namespace, literal_values)
            return json.dumps(result)
        except Exception as e:
            error_message = {"error": str(e)}
            logger.error("Error deploying Kubernetes secret: " + str(e))
            return json.dumps(error_message)


    kubernetes_secret_deployer = StructuredTool.from_function(
        coroutine=kubernetes_secret_deployer_tool,
        name="Kubernetes_Secret_Deployer",
        description="Deploy a Kubernetes secret. Only create this if the cloud service provider is unknown. If the cloud service provider is known, DO NOT create kuberenetes secret.",
        args_schema=KubernetesSecret,
        return_direct=False,
    )


    class HelmChartDeployment(BaseModel):
        namespace: str = Field(
            description="Namespace in which the chart will be deployed")
        auth_method_id: str = Field(
            description="Akeyless access id for auth method akeyless_access_id")
        release_name: str = Field(
            default="gw",
            description="Optional: The release name for the Helm chart. Defaults to 'gw' if not provided.")

    async def helm_chart_deployer_tool(namespace: str, auth_method_id: str, release_name: str = "gw") -> str:
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
            result = await deploy_akeyless_gateway(namespace, auth_method_id, release_name)
            return result
        except Exception as e:
            error_message = {"error": str(e)}
            logger.error(f"Error deploying Helm chart: {e}")
            return json.dumps(error_message)


    helm_chart_deployer = StructuredTool.from_function(
        coroutine=helm_chart_deployer_tool,
        name="Helm_Chart_Deployer",
        description="Deploy the Akeyless Gateway Helm chart in a Kubernetes cluster",
        args_schema=HelmChartDeployment,
        return_direct=False,
    )

    async def can_i_deploy_into_namespace_tool(namespace: str) -> str:
        """
        This tool is used to check if the bot can deploy into the namespace.
        It makes an external API call to check if the bot has the necessary permissions.
        """
        hasDeploymentPermission = await can_i_deploy_into_namespace(namespace)
        return json.dumps({"can_i_deploy": hasDeploymentPermission})


    can_i_deploy_into_namespace_checker = StructuredTool.from_function(
        coroutine=can_i_deploy_into_namespace_tool,
        name="Can_I_Deploy_Into_Namespace",
        description="Check if the bot has permission to deploy the helm chart into the namespace. You can only run this tool AFTER you have detected the namespace and service account. If the bot does not have permission to deploy into the namespace, then do not deploy the helm chart.",
        return_direct=False,
    )


    async def get_list_of_helm_releases_in_namespace_tool(namespace: str) -> str:
        """
        This tool is used to get the list of helm releases in a namespace.
        It makes an external API call to get the list of releases.
        """
        helm_releases_in_namespace: List[str] = get_deployed_helm_releases(namespace)
        return json.dumps({"helm_releases_in_namespace": helm_releases_in_namespace})


    get_list_of_helm_releases_in_namespace = StructuredTool.from_function(
        coroutine=get_list_of_helm_releases_in_namespace_tool,
        name="Get_List_Of_Helm_Releases_In_Namespace",
        description="Get the list of helm releases in a namespace. This tool can be used to determine if the anticipated helm chart release name is already taken and a new name needs to be generated and used.",
        return_direct=False,
    )


    class TokenValidation(BaseModel):
        token: str = Field(description="The Akeyless token to be validated")


    async def get_akeyless_token_validation_information_tool(token: str) -> str:
        """
        This tool is used to get the details of the validation result of an Akeyless token.
        It makes an external API call to get the validation result details.
        
        Args:
            token (str): The Akeyless token to be validated.
        
        Returns:
            str: A JSON string containing the validation result details, including:
                - expiration (str): The expiration time of the token.
                - is_valid (bool): A boolean indicating whether the token is valid.
                - reason (str): The reason for the token's validation status.
        """
        logging.info("Getting Akeyless token validation information.")
        validation_result = await validate_akeyless_token(token)
        logging.debug(f"Validation result: {validation_result}")
        return json.dumps({
            "expiration": validation_result.expiration,
            "is_valid": validation_result.is_valid,
            "reason": validation_result.reason
        })


    get_akeyless_token_validation_information = StructuredTool.from_function(
        coroutine=get_akeyless_token_validation_information_tool,
        name="Get_Akeyless_Token_Validation_Information",
        description="""Get the details of the validation result of an Akeyless token. This tool can be used to determine if the Akeyless token is valid and the reason for its validation status. The result is a JSON string with the following structure: {"is_valid":true,"expiration":"2024-03-29 06:34:40 +0000 UTC"}""",
        args_schema=TokenValidation,
        return_direct=False,
    )



    class AkeylessAuthMethodValidation(BaseModel):
        auth_method_name: str = Field(description="The name of the Akeyless authentication method to be validated.")


    async def check_if_akeyless_auth_method_exists_tool(auth_method_name: str) -> str:
        """
        This tool is used to check if an Akeyless authentication method with a specific name already exists.

        Args:
            auth_method_name (str): The name of the Akeyless authentication method to be validated.

        Returns:
            str: A JSON string indicating whether the authentication method exists and another name should be chosen.
        """
        try:
            logging.info(f"Checking if Akeyless authentication method with name {auth_method_name} exists.")
            auth_method_exists = await check_if_akeyless_auth_method_exists_from_list_auth_methods(auth_method_name)
            return json.dumps({"auth_method_exists": auth_method_exists})
        except Exception as e:
            logging.error(f"Exception when checking if Akeyless authentication method exists: {e}")
            raise


    check_if_akeyless_auth_method_exists = StructuredTool.from_function(
        coroutine=check_if_akeyless_auth_method_exists_tool,
        name="Check_If_Akeyless_Auth_Method_Exists",
        description="Check if an Akeyless authentication method with a specific name already exists, if it does then another name should be chosen.",
        args_schema=AkeylessAuthMethodValidation,
        return_direct=False,
    )

    @tool
    async def python_repl(
        code: Annotated[str, "The python code to execute to generate your chart."]
    ):
        """Use this to execute python code. If you want to see the output of a value,
        you should print it out with `print(...)`. This is visible to the user."""
        try:
            result = repl.run(code)
        except BaseException as e:
            return f"Failed to execute. Error: {repr(e)}"
        return f"Succesfully executed:\n```python\n{code}\n```\nStdout: {result}"

    tools = [
        HumanInputChainlit(),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
            coroutine=llm_math_chain.arun,
        ),
        get_pod_namespace_and_service_account,
        can_i_deploy_into_namespace_checker,
        cloud_service_detector,
        aws_auth_method_creator,
        azure_auth_method_creator,
        gcp_auth_method_creator,
        # api_key_auth_method_creator,
        # kubernetes_secret_deployer,
        helm_chart_deployer,
        # get_list_of_helm_releases_in_namespace,
        get_akeyless_token_validation_information,
        check_if_akeyless_auth_method_exists,
        python_repl
    ]


    prompt2 = ChatPromptTemplate(
        input_variables=["agent_scratchpad", "input", "tool_names", "tools"],
        input_types={
            "chat_history": List[
                Union[
                    AIMessage,
                    HumanMessage,
                    ChatMessage,
                    SystemMessage,
                    FunctionMessage,
                    ToolMessage,
                ]
            ],
            "agent_scratchpad": List[
                Union[
                    AIMessage,
                    HumanMessage,
                    ChatMessage,
                    SystemMessage,
                    FunctionMessage,
                    ToolMessage,
                ]
            ],
        },
        messages=[
            SystemMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=[],
                    template="""Assistant is a large language model trained by OpenAI.\n\nAssistant is designed to be able to assist with a wide range 
                            of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. 
                            As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage 
                            in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
                            \n\nAssistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process 
                            and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide 
                            range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it 
                            to engage in discussions and provide explanations and descriptions on a wide range of topics.
                            \n\nOverall, Assistant is a powerful system that can help with a wide range of tasks and provide valuable insights and 
                            information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation
                            about a particular topic, Assistant is here to assist.""",
                )
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["input", "tool_names", "tools"],
                    template='TOOLS\n------\nAssistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:\n\n{tools}\n\nRESPONSE FORMAT INSTRUCTIONS\n----------------------------\n\nWhen responding to me, please output a response in one of two formats:\n\n**Option 1:**\nUse this if you want the human to use a tool.\nMarkdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": string, \\ The action to take. Must be one of {tool_names}\n    "action_input": string \\ The input to the action\n}}\n```\n\n**Option #2:**\nUse this if you can respond directly to the human after tool execution. Markdown code snippet formatted in the following schema:\n\n```json\n{{\n    "action": "Final Answer",\n    "action_input": string \\ You should put what you want to return to user here\n}}\n```\n\nUSER\'S INPUT\n--------------------\nHere is the user\'s input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):\n\n{input}',

                )
            ),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ],
    )

    agent_runnable = create_json_chat_agent(chat_model, tools, prompt2)

    tool_executor = ToolExecutor(tools)

    # @cl.step(type="llm")
    async def run_agent(data):
        agent_outcome = agent_runnable.invoke(data)
        return {"agent_outcome": agent_outcome}

    # @cl.step(type="tool")
    async def execute_tools(data):
        # current_step = cl.context.current_step
        agent_action = data['agent_outcome']
        tool = agent_action.tool
        tool_input = agent_action.tool_input
        
        if tool != "human":
            async with cl.Step(name=tool) as step:
                step.input = tool_input
                step.output = tool_input
                step.root=True
        
        logging.info(f"Executing tool: {tool}, with inputs: {tool_input}")

        output = await tool_executor.ainvoke(agent_action)
        
        if tool != "human":
            step.output = output
        
        data["intermediate_steps"].append((agent_action, str(output)))
        return data


    def should_continue(data):
        if isinstance(data['agent_outcome'], AgentFinish):
            return "end"
        else:
            return "continue"

    workflow = StateGraph(AgentState)


    workflow.add_node("agent", run_agent)
    workflow.add_node("action", execute_tools)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )

    workflow.add_edge("action", "agent")

    app = workflow.compile()
    
    cl.user_session.set("runner", app)


@cl.on_message
async def main(message: cl.Message):
    chat_history = cl.user_session.get("chat_history")  # type: ConversationBufferMemory
    runner = cl.user_session.get("runner")  # type: CompiledGraph
    
    messages = await chat_history.chat_memory.aget_messages()
    # print(messages)
    inputs = {
        "input": message.content,
        "chat_history": messages,
    }
    
    # Create placeholder for response message from AI
    msg = cl.Message(content="")
    
    answer_prefix_tokens=["Final Answer"]
    
    # Create a new instance of the RunnableConfig
    config = RunnableConfig(
        run_name="Heimdal with Tools and UI",
        recursion_limit=50,
        callbacks=[cl.AsyncLangchainCallbackHandler(
            stream_final_answer=True,
            answer_prefix_tokens=answer_prefix_tokens
        )]
    )
    
    async for output in runner.with_config(config).astream(inputs):
        # log_output = ", ".join([f"{key}: {value}" for key, value in output.items()])
        # logger.info(f"Output received: {log_output}")
        if "agent" in output:
            agent = output["agent"]
            if "agent_outcome" in agent:
                agent_outcome = agent["agent_outcome"]
                if isinstance(agent_outcome, AgentFinish):
                    await msg.stream_token(agent_outcome.return_values["output"])


    
    chat_history.chat_memory.add_message(HumanMessage(content=message.content))
    chat_history.chat_memory.add_message(AIMessage(content=msg.content))
    cl.user_session.set("chat_history", chat_history)
    await msg.send()

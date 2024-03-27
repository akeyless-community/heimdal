from chainlit.input_widget import TextInput
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.messages import AIMessage
import chainlit as cl
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable.config import RunnableConfig
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage
from langchain.memory import ConversationBufferMemory
import os
import logging
from app import app_interactor
from dotenv import load_dotenv
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Load .env file
load_dotenv(verbose=True)

from langchain_openai import ChatOpenAI
from app import AppInteractor, Response

async def ask_helper(func, **kwargs):
    res = await func(**kwargs).send()
    while not res:
        res = await func(**kwargs).send()
    return res

DEPLOY_GATEWAY_COMMAND = "Please check what the kubernetes namespace and service account are for this bot to deploy the akeyless gateway helm chart. Please make sure the bot has permission to deploy the helm chart into this namespace. Please figure out the cloud service provider where this bot is installed and create the appropriate authentication method. Please deploy the helm chart using the authentication method access ID. Once the helm chart is deployed, please return all the details about everything created."


# def setup_runnable():
#     memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    
#     model = ChatOpenAI(streaming=True)
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful chatbot"),
#             MessagesPlaceholder(variable_name="history"),
#             ("human", "{question}"),
#         ]
#     )

#     runnable = (
#         RunnablePassthrough.assign(
#             history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
#         )
#         | prompt
#         | model
#         | StrOutputParser()
#     )
    
#     cl.user_session.set("runnable", runnable)

@cl.author_rename
def rename(orig_author: str):
    # Rename the author of the message
    rename_dict = {"Chatbot": "Heimdal"}
    # Return the renamed author if it exists in the dictionary, otherwise return the original author
    return rename_dict.get(orig_author, orig_author)

@cl.action_callback("Approve the Scanning of the environment and install the Akeyless Gateway")
async def on_action(action):
    # Retrieve the chat memory from the user session
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    # Add the DEPLOY_GATEWAY_COMMAND to the message history as if the user sent it
    await cl.Message(
        content=DEPLOY_GATEWAY_COMMAND,
        type="USER_MESSAGE"
        ).send()
    
    # memory.chat_memory.add_user_message(DEPLOY_GATEWAY_COMMAND)
    
    # Store the updated memory in the user session
    cl.user_session.set("memory", memory)
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
        # If the token is valid, set it as an environment variable
        os.environ["AKEYLESS_TOKEN"] = akeyless_token
        
        # Create an action button to scan the environment and deploy the Akeyless Gateway
        actions = [
            cl.Action(
                name="Approve the Scanning of the environment and install the Akeyless Gateway",
                description="Approve the Scanning of the environment and install the Akeyless Gateway", 
                value=akeyless_token)
        ]
        # Inform the user of the successful token setup
        await cl.Message(content="Akeyless Token set successfully. If you approve the scanning of the environment and wish for me to install the Akeyless Gateway, please click the button and then copy the text.", actions=actions).send()
        
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
    memory = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("memory", memory)
    
    # Create an instance of the ChainlitCallbackHandler
    # callback_handler = ChainlitCallbackHandler()

    # Create the TaskList
    task_list = cl.TaskList()
    cl.user_session.set("task_list", task_list)
    
    # Create an instance of the AppInteractor
    app_interactor.add_callback_handler_and_compile(cl, None, task_list)

    # Store the app_interactor instance in the user session
    cl.user_session.set("app_interactor", app_interactor)


@cl.on_message
async def main(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    
    task_list = cl.user_session.get("task_list") # type: cl.TaskList
    # Retrieve the app_interactor instance from the user session
    app_interactor:AppInteractor = cl.user_session.get("app_interactor")  # type: AppInteractor
    
    # Add the user message to the chat memory
    memory.chat_memory.add_user_message(message.content)
    
    # Store the memory in the user session
    cl.user_session.set("memory", memory)
    
    messages = await memory.chat_memory.aget_messages()
    
    # serialize the messages and print them
    print("Messages from chat memory:", messages)

    # Process the user input
    inputs = {
        "messages": messages,
    }
    
    # Create placeholder for response message from AI
    msg = cl.Message(content="")
    
    # Create a new instance of the RunnableConfig
    config = RunnableConfig(
        run_name="Heimdal with Tools and UI",
        recursion_limit=50,
    )

    # Stream the events from the app_interactor
    async for chunk in app_interactor.app.with_config(config).astream(inputs):
        for key, value in chunk.items():
            print(f"Key: {key}, Value: {value}")
            if key == "agent":
                latest_message: AIMessage = value.get("messages", [])[-1] if value.get("messages") else None
                if latest_message and latest_message.additional_kwargs.get("tool_calls"):
                    for tool_call in latest_message.additional_kwargs["tool_calls"]:
                        if tool_call.get("type") == "function":
                            arguments = tool_call.get("function", {}).get("arguments", "")
                            function_name = tool_call.get("function", {}).get("name", "")
                            message_to_print = f"Running the {function_name} tool"
                            if arguments and arguments != "{}":
                                message_to_print += f" with the arguments {arguments}"
                            else:
                                message_to_print += " with no arguments"
                            print(message_to_print)
                            # message_placeholder.markdown(message_to_print)
                if latest_message:
                    await msg.stream_token(latest_message.content)

            if key == "messages":
                for message in value:
                    await msg.stream_token(message["content"])
                    # st.session_state.messages.append({"role": message["role"], "content": message["content"]})
    await msg.send()
    
    # Add the AI message to the chat memory
    memory.chat_memory.add_ai_message(msg.content)
    
    # Store the memory in the user session
    cl.user_session.set("memory", memory)

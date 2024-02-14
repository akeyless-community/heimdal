# Import necessary libraries
import streamlit as st
import asyncio
# Import Langchain modules
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables.config import RunnableConfig
# Streamlit UI Callback
from langchain_community.callbacks import StreamlitCallbackHandler
from app import app_interactor

# Import modules related to streaming response
import os

# Set up the Streamlit app
st.title("Heimdal - The Agentic Installer")

# Add a sidebar to input the user's Akeyless Token
st.sidebar.header("Akeyless Token")
akeyless_token = st.sidebar.text_input("Enter your Akeyless Token", type="password", help="You can retrieve the token from the Akeyless web console by clicking the top right hand corner down arrow and then choosing 'Copy token'")
# Set the environment variable of AKEYLESS_TOKEN only after setting the token by the user
if akeyless_token:
    if not akeyless_token.startswith("t-"):
        st.error("Akeyless Token must start with 't-'")
    else:
        os.environ["AKEYLESS_TOKEN"] = akeyless_token

view_messages = st.expander("View the message contents in session state")

# Get the API key from the secrets manager
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Check if the button was previously clicked
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False


async def handle_click_async(message_placeholder):
    # Display the user's question in the chat message container
    human_message = HumanMessage(content="Please check what the kubernetes namespace and service account are for this bot to deploy the akeyless gateway helm chart. Please make sure the bot has permission to deploy the helm chart into this namespace. Please figure out the cloud service provider where this bot is installed and create the appropriate authentication method. Please deploy the helm chart using the authentication method access ID. Once the helm chart is deployed, please return all the details about everything created.")

    # Add the user's question to the chat history
    st.session_state.messages.append({"role": "user", "content": human_message.content})

    # Set up the Streamlit callback handler
    inputs = {
        "messages": st.session_state.messages
    }
    with st.spinner('Processing...'):
        st.session_state.messages.append({"role": "assistant", "content": "Installation process started."})
        for event in app_interactor.app.with_config({"run_name": "Heimdal with Tools and UI"}).stream(inputs):
            for key, value in event.items():
                if key == "agent":
                    latest_message: AIMessage = value["messages"][-1]
                    if latest_message.additional_kwargs["tool_calls"][0]["type"] == "function":
                        arguments = latest_message.additional_kwargs["tool_calls"][0]["function"]["arguments"]
                        function_name = latest_message.additional_kwargs["tool_calls"][0]["function"]["name"]
                        if arguments == "{}":
                            message_placeholder.markdown(f"Running the {function_name} tool with no arguments")
                        else:
                            message_placeholder.markdown(f"Running the {function_name} tool with the arguments {arguments}")

                if key == "messages":
                    for message in value:
                        st.session_state.messages.append({"role": message["role"], "content": message["content"]})

    st.success('Done!')

def handle_click(message_placeholder):
    # Change the state to reflect the button was clicked
    st.session_state.button_clicked = True
    # Run the async function
    asyncio.run(handle_click_async(message_placeholder))

# Initialize chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": """
                  Hello, I'm Heimdal 👋 A prototype "Agentic" Helm Chart Installer. 
                  
                  I can help you install and configure the Akeyless Gateway into the very namespace I am currently deployed into, provided you give me the necessary permissions. 
                  
                  I am able to run tools like scanning the environment to find out the kubernetes namespace and service account of this deployment.
                  
                  I can also figure out the cloud service provider where this bot is installed and create the appropriate Akeyless authentication method.
                  
                  I can deploy the helm chart using the authentication method Access ID. Once the helm chart is deployed, I can return all the details about everything we've created.
                  
                  Thank you for using The Akeyless "Vaultless" Platform!
                  
                  With your approval, I will scan the environment and install the Akeyless Gateway. 
                  
                  Please click the red button to approve the installation.
                  
                  
                  """}
    ]


if akeyless_token and akeyless_token.startswith("t-"):

    # Check if the button was previously clicked. Only show the button if it hasn't been clicked yet
    if not st.session_state.button_clicked:
        st.chat_message("assistant").write(st.session_state.messages[0]["content"])
        st_callback = StreamlitCallbackHandler(st.container())
        message_placeholder = st.empty()
        # Process the user's question and generate a response
        if st.button("Approve the Scanning of the environment and install the Akeyless Gateway", type="primary"):
            handle_click(message_placeholder)
    else:
        # Display previous chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

else:
    st.warning("Please populate the Akeyless Token before proceeding.")
    
# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.messages)
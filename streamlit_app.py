# Import necessary libraries
import streamlit as st

# Import Langchain modules
from langchain.agents.tools import Tool
from langchain import OpenAI
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.chains import LLMChain
# Streamlit UI Callback
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import LLMMathChain
from langchain.memory import ConversationBufferMemory
from app import app_interactor

import openai


# Import modules related to streaming response
import os
import time


# ---
# Set up the Streamlit app
st.title("Heimdal - The Prototype Agentic Installer")



st.write(
    """
    👋 Welcome to Heimdal the 🌎
"""
)


# Get the API key from the secrets manager
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize chat history if it doesn't already exist
if "messages" not in st.session_state:
    st.session_state.messages = []


# Initialize the OpenAI language model and search tool

# llm = OpenAI(temperature=0)
# llm_math_chain = LLMMathChain(llm=llm, verbose=True)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")


# Display previous chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process the user's question and generate a response
if st.button("Approve the Scanning of the environment and install the Akeyless Gateway"):
    # Display the user's question in the chat message container
    human_message = HumanMessage(content="Please check what the kubernetes namespace and service account are for this bot to deploy the akeyless gateway helm chart. Please make sure the bot has permission to deploy the helm chart into this namespace. Please figure out the cloud service provider where this bot is installed and create the appropriate authentication method. Please deploy the helm chart using the authentication method access ID. Once the helm chart is deployed, please return all the details about everything created.")
    
    with st.chat_message("user"):
        st.markdown(human_message.content)

    # Add the user's question to the chat history
    st.session_state.messages.append(human_message)

    # Generate the assistant's response
    with st.chat_message("assistant"):
        # Set up the Streamlit callback handler
        st_callback = StreamlitCallbackHandler(st.container())
        message_placeholder = st.empty()
        full_response = ""
        # assistant_response = conversational_agent.run(question, callbacks=[st_callback])
        inputs = {
            "messages": st.session_state.messages
        }
        assistant_response = app_interactor.app.with_config({"run_name": "Heimdal with Tools and UI"}).stream(inputs, callbacks=[st_callback])
        
        # Simulate a streaming response with a slight delay
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)

            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        # Display the full response
        message_placeholder.info(full_response)

    # Add the assistant's response to the chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

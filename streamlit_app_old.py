import streamlit as st
import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


st.set_page_config(page_title="Heimdal Chatbot", page_icon="📖")
st.title(" Heimdal Chatbot")

"""
A basic example of using StreamlitChatMessageHistory to help LLMChain remember messages in a conversation.
The messages are stored in Session State across re-runs automatically. You can view the contents of Session State
in the expander below. View the
[source code for this app](https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/basic_memory.py).
"""

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Sidebar for Akeyless token input
st.sidebar.title("Akeyless Authentication")
token = st.sidebar.text_input(
    "Enter your Akeyless token:",
    type="password",
    help="Find your Akeyless token by logging into any Akeyless web console, "
         "click the down arrow in the top right-hand corner, and then click 'Copy token'."
)

# Validating the token format
if token and not token.startswith("t-"):
    st.sidebar.error("Invalid token format. Must start with 't-'.")

# Set up the LangChain, passing in Message History

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain = prompt | ChatOpenAI()
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    msgs.add_user_message(prompt)
    st.chat_message("human").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        response = llm.invoke(prompt)
        msgs.add_ai_message(response.content)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
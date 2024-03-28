from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import BaseMessage
from langchain.chains.llm_math.base import LLMMathChain
import operator
import chainlit as cl
from chainlit.sync import run_sync
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langchain.agents import create_json_chat_agent, Tool
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, PromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.agents import create_json_chat_agent
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


@cl.on_chat_start
async def start():
    # Create a new instance of the chat memory
    chat_history = ConversationBufferMemory(return_messages=True)
    cl.user_session.set("chat_history", chat_history)

    repl = PythonREPL()
    
    

    chat_model = ChatOpenAI(
            temperature=0,
            streaming=True,
            model="gpt-3.5-turbo"
        )

    llm_math_chain = LLMMathChain.from_llm(llm=chat_model, verbose=True)

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

        output = tool_executor.invoke(agent_action)
        
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
                
                if isinstance(agent_outcome, AgentAction):
                    if agent_outcome.tool != "human":
                        print(f"AgentAction outcome output: {agent_outcome.return_values['output']}")

    
    chat_history.chat_memory.add_message(HumanMessage(content=message.content))
    chat_history.chat_memory.add_message(AIMessage(content=msg.content))
    cl.user_session.set("chat_history", chat_history)
    await msg.send()

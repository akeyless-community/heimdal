# Initialize the appropriate LLMs using environment variables
# instructor.patch enables response_model keyword
# import instructor
import os
from langchain_openai import ChatOpenAI
# Import the correct module based on the new LangChain library structure
# from langchain_community.llms import ChatOpenAI  # The new import from langchain-community


llm_heavy = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4-1106-preview")
llm_light = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5-turbo-1106")
# llm_heavy = instructor.patch(ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-4-1106-preview"))
# llm_light = instructor.patch(ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), model="gpt-3.5-turbo-1106"))

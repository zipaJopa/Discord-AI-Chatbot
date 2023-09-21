from langchain.agents import initialize_agent, AgentType, Tool
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from dotenv import find_dotenv, load_dotenv
import openai
import os

load_dotenv(find_dotenv())
openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [        
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),        
]

# messages = [
#         {"role": "system", "name": "instructions", "content": "you can ask me questions about math"},
#         {"role": "user", "content": "I'm Jason, how are you?"},
#         {"role": "assistant", "content": "I'm doing well, how are you?"},
#     ]

messages = """
USER: I'm Jason, how are you?
ASSISTANT: I'm doing well, how are you?
USER: what's my name?
"""

agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, chat_history = messages)

response = agent.run(messages)

print(response)
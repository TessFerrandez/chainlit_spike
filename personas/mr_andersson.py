import os

from dotenv import find_dotenv, load_dotenv
from langchain.agents import initialize_agent, Tool
from langchain.chains import LLMMathChain
from langchain_community.utilities.bing_search import BingSearchAPIWrapper
from langchain_openai import AzureChatOpenAI
from personas.persona import Persona

import chainlit as cl


load_dotenv(find_dotenv())


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")


class MrAndersson(Persona):
    async def on_chat_start(self):
        await cl.Message(content="Hello! I'm Mr.Andersson! Secret agent, I know news and mathz!").send()
        llm = AzureChatOpenAI(
            temperature=0.5,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            streaming=True
        )
        search = BingSearchAPIWrapper()
        llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events. You should ask targeted questions"
            ),
            Tool(
                name="Calculator",
                func=llm_math_chain.run,
                description="useful for when you need to answer questions about math",
            )
        ]
        self.agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)

    async def on_message(self, message):
        cb = cl.LangchainCallbackHandler(stream_final_answer=True)
        await cl.make_async(self.agent.run)(message.content, callbacks=[cb])

"""
For internal LLM use: 

        AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
        AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        API_KEY = os.getenv("AZURE_OPENAI_KEY")
        BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")
        MODEL_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
        DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

        llm = AzureChatOpenAI(
            azure_endpoint=BASE_URL,
            api_key=API_KEY,
            api_version=MODEL_VERSION,
            model=DEPLOYMENT,
            temperature=0.5,
            callbacks=[OpenAICallbackHandler()]
        )
"""

import os

from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI
from personas.persona import Persona

import chainlit as cl

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

API_KEY = os.getenv("AZURE_OPENAI_KEY")
BASE_URL = os.getenv("AZURE_OPENAI_ENDPOINT")
MODEL_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

class ChattyMcChatface(Persona):
    async def on_chat_start(self):
        actions = [
            cl.Action(name="action_button", value="example_value", description="Click me!")
        ]
        llm = AzureChatOpenAI(
            azure_endpoint=BASE_URL,
            api_key=API_KEY,
            api_version=MODEL_VERSION,
            model=DEPLOYMENT,
            temperature=0.5,
            callbacks=[OpenAICallbackHandler()]
        )
        self.chain = ConversationChain(
            llm=llm,
            output_parser=StrOutputParser(),
            callbacks=[OpenAICallbackHandler()],
            verbose=True,
            memory=ConversationBufferMemory(),
        )
        await cl.Message(content="Hello! I'm Chatty McChatface. I'm here to chat with you about anything you want!").send()

    async def on_message(self, message: cl.Message):
        actions = [
            cl.Action(name="action_button", value="example_value", description="Click me!")
        ]
        output_message = cl.Message(content="", actions=actions)

        with get_openai_callback() as cb:
            result = self.chain.invoke({"input": message.content})
            output_message.content = result["response"]
            print(cb)
            tokens = cb.total_tokens

        await output_message.send()
        await cl.Message(content=f"You spent {tokens} tokens").send()

    @cl.action_callback("action_button")
    async def on_action(action):
        chain = cl.user_session.get("persona").chain
        output_message = cl.Message(content="")
        query = chain.memory.chat_memory.messages[-2].content

        chain.memory.chat_memory.messages[-2:] = []
        with get_openai_callback() as cb:
            result = chain.invoke({"input": query})
            output_message.content = result["response"]
            tokens = cb.total_tokens

        await output_message.send()
        await cl.Message(content=f"You spent {tokens} tokens").send()

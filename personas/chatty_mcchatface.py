import os

from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import ConversationChain
from langchain_openai import AzureChatOpenAI
from personas.persona import Persona

import chainlit as cl


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")


class ChattyMcChatface(Persona):
    async def on_chat_start(self):
        llm = AzureChatOpenAI(
            temperature=0.5,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            callbacks=[OpenAICallbackHandler()]
        )
        self.chain = ConversationChain(
            llm=llm,
            output_parser=StrOutputParser(),
            callbacks=[OpenAICallbackHandler()],
        )
        await cl.Message(content="Hello! I'm Chatty McChatface. I'm here to chat with you about anything you want!").send()

    async def on_message(self, message: cl.Message):
        output_message = cl.Message(content="")

        with get_openai_callback() as cb:
            result = self.chain.invoke({ "input": message.content })
            output_message.content = result["response"]
            print(cb)
            tokens = cb.total_tokens

        await output_message.send()
        await cl.Message(content=f"You spent {tokens} tokens").send()

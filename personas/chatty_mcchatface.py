import os

from langchain.prompts import ChatPromptTemplate
from langchain_community.callbacks import get_openai_callback
from langchain.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.output_parsers import StrOutputParser
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
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a very chatty and helpful AI",
                ),
                ("human", "{question}"),
            ]
        )
        self.chain = prompt | llm | StrOutputParser()
        await cl.Message(content="Hello! I'm Chatty McChatface. I'm here to chat with you about anything you want!").send()

    async def on_message(self, message: cl.Message):
        output_message = cl.Message(content="")

        with get_openai_callback() as cb:
            output_message.content = self.chain.invoke({ "question": message.content })
            print(cb)
            tokens = cb.total_tokens

        await output_message.send()
        await cl.Message(content=f"You spent {tokens} tokens").send()

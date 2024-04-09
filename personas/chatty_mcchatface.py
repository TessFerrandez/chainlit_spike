import os

from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig
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
            streaming=True
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
        self.runnable = prompt | llm | StrOutputParser()
        await cl.Message(content="Hello! I'm Chatty McChatface. I'm here to chat with you about anything you want!").send()

    async def on_message(self, message: cl.Message):
        output_message = cl.Message(content="")

        async for chunk in self.runnable.astream(
            { "question": message.content },
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])
        ):
            await output_message.stream_token(chunk)
        await output_message.send()

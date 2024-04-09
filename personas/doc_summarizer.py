import os

import chainlit as cl

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_openai import AzureChatOpenAI
from personas.persona import Persona


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")


class DocSummarizer(Persona):
    async def summarize_doc(self):
        files = None

        while not files:
            files = await cl.AskFileMessage(
                content="Please upload a text file to begin!",
                accept=["text/plain"],
                max_size_mb=1,
                timeout=180
            ).send()

        file = files[0]
        message = cl.Message(content=f"Processing {file.name}...", disable_feedback=True)
        await message.send()
        docs = TextLoader(file.path, encoding='utf-8').load()

        summary = self.summarize_chain.invoke(docs)
        message.content = f"Heres the summary: {summary["output_text"]}"
        await message.update()


    async def on_chat_start(self):
        await cl.Message(content="Hello! I'm Doc Summarizer. I'm here to summarize your docs!").send()
        llm = AzureChatOpenAI(
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.0,
        )
        self.summarize_chain = load_summarize_chain(llm, chain_type="stuff")
        await self.summarize_doc()


    async def on_message(self, message):
        await self.summarize_doc()

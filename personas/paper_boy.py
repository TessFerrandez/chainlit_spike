import os

import chainlit as cl

from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers.arxiv import ArxivRetriever
from langchain_openai import AzureChatOpenAI
from personas.persona import Persona


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")


class PaperBoy(Persona):
    async def summarize_doc(self, topic):
        retriever = ArxivRetriever(load_max_docs=1)
        docs = retriever.get_relevant_documents(query=topic)
        print(docs[0])
        summary = self.summarize_chain.invoke(docs)
        await cl.Message(f"Heres the summary: {summary["output_text"]}").send()


    async def on_chat_start(self):
        await cl.Message(content="Hello! I'm Paper Boy. Gimme a URL plz!").send()
        llm = AzureChatOpenAI(
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.0,
        )
        self.summarize_chain = load_summarize_chain(llm, chain_type="stuff")


    async def on_message(self, message):
        await self.summarize_doc(message.content)

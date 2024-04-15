import os

import chainlit as cl

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents.base import Document
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter
import validators
from personas.persona import Persona


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")


prompt_template = """I want you to act as a research paper summarizer for a {level} audience.

                        I will provide you with a research paper on a specific topic. 
                        You will create a summary of the main points in the paper. You should include any personal opinions or interpretations.
                        The summary should be approximately 1000 words long.

                        The text from the paper is:

                        {text}"""

# prompt_template = """wow this is an easy prompt: {text}"""

levels = {"beginner": "primary school",
          "expert": "PhD student",}

PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"]) 


class PaperBoy(Persona):
    async def summarize_doc(self, url):
        level = "intermediate"
        is_url = validators.url(url)
        if not is_url:
            await cl.Message(f"{url} is not a valid url... Try again").send()
            return

        loader = PyPDFLoader(url)
        docs = loader.load()

        text = "".join([doc.page_content for doc in docs])
        await cl.Message(f"Got the doc from {url}!!").send()
        total = len(docs)
        summary = self.summarize_chain.invoke({ "text": Document(page_content=text), "level": levels["beginner"]})
        await cl.Message(f"Heres the summary: {summary}").send()

    async def on_chat_start(self):
        await cl.Message(content="Hello! I'm Paper Boy. Gimme a topic!").send()
        llm = AzureChatOpenAI(
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.0,
        )
        self.summarize_chain = PROMPT | llm | StrOutputParser()

    async def on_message(self, message):
        await self.summarize_doc(message.content)

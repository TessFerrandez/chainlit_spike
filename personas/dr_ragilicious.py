import os
import chainlit as cl

from dotenv import find_dotenv, load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from personas.persona import Persona


load_dotenv(find_dotenv())


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")


class DrRAGilicious(Persona):
    @staticmethod
    async def process_file(file):
        with open(file.path, "r", encoding="utf-8") as f:
            text = f.read()

        # split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(text)

        # create a metadata for each chunk
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        # create a Chroma vector store
        embeddings = AzureOpenAIEmbeddings(
            deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT,
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION
        )
        doc_search = await cl.make_async(Chroma.from_texts)(texts, embeddings, metadatas=metadatas)
        return doc_search

    @staticmethod
    def create_conversation_buffer():
        message_history = ChatMessageHistory()
        return ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True
        )

    async def on_chat_start(self):
        await cl.Message(content="Hello! I'm Dr. RAGilicious. I'll look at your docs and we can talk about them!").send()

        files = None

        while not files:
            files = await cl.AskFileMessage(
                content="Please upload a text file to begin!",
                accept=["text/plain"],
                max_size_mb=20,
                timeout=180
            ).send()

        file = files[0]
        msg = cl.Message(content=f"Processing {file.name}...", disable_feedback=True)
        await msg.send()

        doc_search = await self.process_file(file)
        memory = self.create_conversation_buffer()
        llm = AzureChatOpenAI(
            api_version=AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION,
            deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT,
            temperature=0.0,
        )

        # create a chain that uses the Chroma vector store
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=doc_search.as_retriever(),
            memory=memory,
            return_source_documents=True
        )

        # Let the user know we're ready to chat
        msg.content = f"Processing {file.name} done! Ready to chat!"
        await msg.update()

    async def on_message(self, message):
        cb = cl.AsyncLangchainCallbackHandler()

        result = await self.chain.ainvoke(message.content, callbacks=[cb])
        answer = result["answer"]
        source_documents = result["source_documents"]

        text_elements = []

        if source_documents:
            for source_idx, source_doc in enumerate(source_documents):
                source_name = f"source_{source_idx}"
                text_elements.append(cl.Text(content=source_doc.page_content, name=source_name))

            source_names = [text_element.name for text_element in text_elements]

            if source_names:
                answer += f"\nSources: {', '.join(source_names)}"
            else:
                answer += "\nNo sources found."

        await cl.Message(content=answer, elements=text_elements).send()

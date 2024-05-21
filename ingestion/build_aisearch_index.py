import os
from typing import Callable

from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_core.embeddings import Embeddings

from azure.search.documents.indexes.models import (
    FreshnessScoringFunction,
    FreshnessScoringParameters,
    ScoringProfile,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    TextWeights,
)

from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv(".env"))


AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT")

AZURE_AISEARCH_ENDPOINT = os.getenv("AZURE_AISEARCH_ENDPOINT")
AZURE_AISEARCH_KEY = os.getenv("AZURE_AISEARCH_KEY")

class AzureSearchVectorStoreBuilder():
    def __init__(self, index_name: str, embedding_function: Callable | Embeddings):
        self.index_name = index_name
        self.get_client(embedding_function)
        
    def get_client(self, embedding_function: Callable | Embeddings):
        fields = [
            SimpleField(
                name="id",
                type=SearchFieldDataType.String,
                key=True,
                filterable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=len(embedding_function("Text")),
                vector_search_profile_name="myHnswProfile",
            ),
            SearchableField(
                name="metadata",
                type=SearchFieldDataType.String,
                searchable=True,
            ),
            # Additional field for filtering on document source
            SimpleField(
                name="source",
                type=SearchFieldDataType.String,
                filterable=True,
            ),
            # Additional data field for last doc update
            SimpleField(
                name="last_update",
                type=SearchFieldDataType.DateTimeOffset,
                searchable=True,
                filterable=True,
            ),
        ]
        # Adding a custom scoring profile with a freshness function
        sc_name = "scoring_profile"
        sc = ScoringProfile(
            name=sc_name,
            text_weights=TextWeights(weights={"content": 5}),
            function_aggregation="sum",
            functions=[
                FreshnessScoringFunction(
                    field_name="last_update",
                    boost=100,
                    parameters=FreshnessScoringParameters(boosting_duration="P2D"),
                    interpolation="linear")
                    ],
        )
        self.client: AzureSearch = AzureSearch(
            azure_search_endpoint=AZURE_AISEARCH_ENDPOINT,
            azure_search_key=AZURE_AISEARCH_KEY,
            index_name=self.index_name,
            embedding_function=embedding_function,
            fields=fields,
            scoring_profiles=[sc],
            default_scoring_profile=sc_name,
        )
    def build_index(self, docs_dir: str, chunk_size: int = 4000, chunk_overlap: int = 0, language_type: Language = Language.HTML):

        text_splitter = RecursiveCharacterTextSplitter.from_language(language=language_type, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for file_content in  DirectoryLoader(docs_dir, glob="*.html", show_progress=True, use_multithreading=True, loader_cls=TextLoader).load():
            docs = text_splitter.split_documents([file_content])
            self.client.add_documents(docs)
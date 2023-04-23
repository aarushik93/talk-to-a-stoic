from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import os
import weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredEPubLoader


def check_batch_result(results: dict):
    """
    Check batch results for errors.
    Parameters
    ----------
    results : dict
        The Weaviate batch creation return value.
    """

    if results is not None:
        for result in results:
            if "result" in result and "errors" in result["result"]:
                if "error" in result["result"]["errors"]:
                    print(result["result"])

def ingest():

    loader = UnstructuredEPubLoader("books/letters.epub",mode="elements")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(pages)

    WEAVIATE_URL = "https://stoicism-0fx6qgtt.weaviate.network"
    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={
            'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"],
        }
    )

    client.schema.delete_all()
    client.schema.get()
    schema = {
        "classes": [
            {
                "class": "Paragraph",
                "description": "A written paragraph",
                "vectorizer": "text2vec-openai",
                "moduleConfig": {
                    "text2vec-openai": {
                        "model": "curie",
                        "modelVersion": "001",
                        "type": "text"
                    }
                },
                "properties": [
                    {
                        "dataType": ["text"],
                        "description": "The content of the paragraph",
                        "moduleConfig": {
                            "text2vec-openai": {
                                "skip": False,
                                "vectorizePropertyName": False
                            }
                        },
                        "name": "content",
                    },
                ],
            },
        ]
    }

    client.schema.create(schema)

    print("Ingesting data...stay tuned")

    client.batch(
        batch_size=100,
        dynamic=True,
        creation_time=5,
        timeout_retries=3,
        connection_error_retries=3,
        callback=check_batch_result,
    )

    with client.batch as batch:
        for text in documents:
            batch.add_data_object(
                {"content": text.page_content, "source": str(text.metadata["source"])},
                "Paragraph",
            )

ingest()
"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chains import ConversationalRetrievalChain, ChatVectorDBChain, RetrievalQAWithSourcesChain
import weaviate
import os
from langchain.vectorstores.weaviate import Weaviate

def get_chain( question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """Create a ChatVectorDBChain for question/answering."""
    # Construct a ChatVectorDBChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])

    OPENAI_API_KEY = "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"
    WEAVIATE_URL = "https://stoicism-0fx6qgtt.weaviate.network"  # books live here

    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={
            'X-OpenAI-Api-Key': OPENAI_API_KEY
        }
    )

    vectorstore = Weaviate(client, "Paragraph", "content")
    ret = vectorstore.as_retriever()

    AI = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

    qa = ConversationalRetrievalChain.from_llm(
        AI,
        ret,
        callback_manager=manager,
    )

    return qa

from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain, ChatVectorDBChain, RetrievalQAWithSourcesChain
import weaviate
import os



def start_bot():
    OPENAI_API_KEY = "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"
    WEAVIATE_URL = "https://stoicism-0fx6qgtt.weaviate.network"  # books live here

    client = weaviate.Client(
        url=WEAVIATE_URL,
        additional_headers={
            'X-OpenAI-Api-Key': os.environ["OPENAI_API_KEY"]
        }
    )

    vectorstore = Weaviate(client, "Paragraph", "content")
    ret = vectorstore.as_retriever()

    AI = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

    # qa = ChatVectorDBChain.from_llm(AI, vectorstore)
    qa = ConversationalRetrievalChain.from_llm(AI, ret)
    # qa = ConversationalRetrievalChain.from_llm(AI, ret, return_source_documents=True)

    chat_history = []

    while True:
        query = input("")
        result = qa({"question": query, "chat_history": chat_history})
        print(result["answer"])
        chat_history = [(query, result["answer"])]


start_bot()
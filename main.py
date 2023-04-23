import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import weaviate
import os
from langchain.utilities import GoogleSerperAPIWrapper


# Set up the Streamlit app
st.title("Talk to a Modern Stoic")

SERPER_API_KEY = "824b5e0d97973ee3e0d750e0f572d1773ab8a031"
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

# Set up the question-answering system
qa = RetrievalQA.from_chain_type(
    llm=AI,
    chain_type="stuff",
    retriever=ret,
)

search = GoogleSerperAPIWrapper(serper_api_key=SERPER_API_KEY)

# Set up the conversational agent
tools = [
    Tool(
        name="Stoic System",
        func=qa.run,
        description="Useful for getting information rooted in Stoicism. Ask questions based on themes, life issues and feelings ",
    ),
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to get current, up to date answers."
    )
]
prefix = """You are a stoic giving people advice, as best you can based on the context and memory available.
            Your answers should be directed at the human, say "you".
            Add an example, relevant in 2023, to illustrate the meaning of the answer.
            Always apply stoic principles.
            You have access to to two tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history"
    )

llm_chain = LLMChain(
    llm=OpenAI(
        temperature=0.2, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo"
    ),
    prompt=prompt,
)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=st.session_state.memory
)

# Allow the user to enter a query and generate a response
query = st.text_input(
    "What's bothering you today?",
    placeholder="Ask me about life",
)

if query:
    with st.spinner(
            "Thinking...."
    ):
        res = agent_chain.run(query)
        st.info(query, icon="😊")
        st.info(res, icon="🤖")

# Allow the user to view the conversation history and other information stored in the agent's memory
with st.expander("My thinking"):
    st.session_state.memory.return_messages




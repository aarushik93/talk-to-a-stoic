import streamlit as st
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent, ReActTextWorldAgent, initialize_agent
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.vectorstores.weaviate import Weaviate
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
import weaviate
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.chat_models import ChatOpenAI

st.title("Talk to a Modern Stoic")

SERPER_API_KEY = "<YOUR_KEY>"
OPENAI_API_KEY = "<YOUR_KEY>"
WEAVIATE_URL = "<YOUR_URL>"  # books live here

client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers={
        'X-OpenAI-Api-Key': OPENAI_API_KEY
    }
)

vectorstore = Weaviate(client, "Paragraph", "content")
ret = vectorstore.as_retriever()

AI = OpenAI(temperature=0.2, openai_api_key=OPENAI_API_KEY)

# Set up the retrieval QA system. This pulls docs from Weaviate and answers questions.
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
prefix = """You are a Stoic giving people advice using Stoicism, based on the context and memory available.
            Add specific examples, relevant in 2023, to illustrate the meaning of the answer.
            You can use these two tools to two tools:"""
suffix = """Start!"
Chat History:
{chat_history}
Latest Question: {input}
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
    llm=ChatOpenAI(
        temperature=0.8, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo"
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
        st.info(res, icon="🤖")

# Optional: Allow users to see the bot's "thinking"
with st.expander("My thinking"):
    st.session_state.memory.return_messages




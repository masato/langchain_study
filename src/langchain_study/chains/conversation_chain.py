from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)

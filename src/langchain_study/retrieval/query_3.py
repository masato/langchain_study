from langchain.chains import RetrievalQA
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

database = Chroma(persist_directory="./data", embedding_function=embeddings)

retriever = database.as_retriever()

qa = RetrievalQA.from_llm(llm=llm, retriever=retriever, return_source_documents=True)

result = qa("飛行車の最高速度は?")
print(result["result"])
print(result["source_documents"])

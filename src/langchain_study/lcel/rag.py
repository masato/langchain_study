"""RAG (Retrieval-Augmented Generation) の例."""

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

texts = [
    "私の趣味は読書です。",
    "私の好きな食べ物はカレーです。",
    "私の嫌いな食べ物は饅頭です。",
]

vectorstore = FAISS.from_texts(
    texts,
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

prompt = ChatPromptTemplate.from_template(
    """以下のcontextだけに基づいて回答してください。

{context}

質問: {question}
""",
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
)
output_parser = StrOutputParser()

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model  #
    | output_parser  # type: ignore[var-annotated]
)

result = chain.invoke("私の好きな食べ物は何でしょう?")
print(result)

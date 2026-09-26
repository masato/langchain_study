import chainlit as cl
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

model = ChatOpenAI(model="gpt-3.5-turbo-1106")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "文章を元に質問に答えてください。\n\n文章: {document}"),
        ("human", "質問: {query}"),
    ],
)

database = Chroma(persist_directory="./data", embedding_function=embeddings)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content="準備ができました。メッセージを入力してください").send()


@cl.on_message
async def on_message(input_message: cl.Message) -> None:
    documents = database.similarity_search(input_message.content)

    documents_string = ""

    for document in documents:
        documents_string += f"""
    ----------------------------
    {document.page_content}
    """

    chain = prompt | model | StrOutputParser()
    result = chain.invoke(
        {"document": documents_string, "query": input_message.content},
    )
    await cl.Message(content=result).send()

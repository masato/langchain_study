from langchain_community.vectorstores.chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

database = Chroma(persist_directory="./data", embedding_function=embeddings)

query = "飛行車の最高速度は?"

documents = database.similarity_search(query)

documents_string = ""

for document in documents:
    documents_string += f"""
----------------------------
{document.page_content}
"""

prompt = PromptTemplate(
    template="""文章を元に答えてください。

文章:
{document}

質問: {query}
""",
    input_variables=["document", "query"],
)

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

result = chat.invoke(
    [HumanMessage(content=prompt.format(document=documents_string, query=query))],
)

print(result.content)

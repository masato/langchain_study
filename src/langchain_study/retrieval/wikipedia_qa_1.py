from langchain.chains import RetrievalQA
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()
retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
    top_k_results=2,
    wiki_client=None,
)

chain = RetrievalQA.from_llm(
    llm=chat,
    retriever=retriever,
    return_source_documents=True,
)

result = chain.invoke(input={"query": "バーボンウイスキーとは?"})
source_documents = result["source_documents"]
print(f"検索結果: {len(source_documents)}件")


for document in source_documents:
    print("------------取得したメタデータ-------------")
    print(document.metadata)
    print("------------取得したテキスト-------------")
    print(document.page_content[:100])

print("------------返答-------------")
print(result["result"])

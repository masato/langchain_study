import wikipedia
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import RePhraseQueryRetriever
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

retriever = WikipediaRetriever(
    wiki_client=wikipedia,
    lang="ja",
    doc_content_chars_max=500,
)

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=PromptTemplate(
        input_variables=["question"],
        template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
        質問: {question}
        """,
    ),
)

retriever_from_llm_chain = RePhraseQueryRetriever(
    retriever=retriever,
    llm_chain=llm_chain,
)

docs = retriever_from_llm_chain.get_relevant_documents(
    "私はラーメンが好きです。ところでバーボンウイスキーとは何ですか?",
)

print(docs)

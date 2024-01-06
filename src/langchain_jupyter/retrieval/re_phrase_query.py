from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.retrievers import RePhraseQueryRetriever, WikipediaRetriever
from langchain_openai import ChatOpenAI

retriever = WikipediaRetriever(lang="ja", doc_content_chars_max=500)

llm_chain = LLMChain(
    llm=ChatOpenAI(temperature=0),
    prompt=PromptTemplate(
        input_variables=["question"],
        template="""以下の質問からWikipediaで検索するべきキーワードを抽出してください。
        質問: {question}
        """,
    ),
)

re_phrase_query_retriever = RePhraseQueryRetriever(
    llm_chain=llm_chain,
    retriever=retriever,
)

documents = re_phrase_query_retriever.get_relevant_documents(
    "私はラーメンが好きです。ところでバーボンウイスキーとは何ですか?",
)

print(documents)

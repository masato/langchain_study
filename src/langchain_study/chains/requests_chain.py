from langchain.chains import LLMChain, LLMRequestsChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

prompt = PromptTemplate(
    input_variables=["query", "requests_result"],
    template="""以下の文章を元に質問に答えてください。
文章: {requests_result}
質問: {query}""",
)

llm_chain = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True,
)

chain = LLMRequestsChain(
    llm_chain=llm_chain,
)

result = chain.invoke(
    {
        "query": "東京の天気について教えてください。",
        "url": "https://www.jma.go.jp/bosai/forecast/data/forecast/130000.json",
    },
)

print(result["output"])

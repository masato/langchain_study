from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

write_article_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate(
        template="{input}について記事を書いてください。",
        input_variables=["input"],
    ),
)

translate_chain = LLMChain(
    llm=chat,
    prompt=PromptTemplate(
        template="以下の文章を英語にしてください。\n\n{input}",
        input_variables=["input"],
    ),
)

sequential_chain = SimpleSequentialChain(
    chains=[write_article_chain, translate_chain],
)

result = sequential_chain.invoke({"input": "エレキギターの選び方"})

print(result["output"])

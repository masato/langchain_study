from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-3.5-turbo")

write_article_prompt = ChatPromptTemplate.from_template(
    "{input}について記事を書いてください.",
)

translate_prompt = ChatPromptTemplate.from_template(
    "以下の文章を英語にしてください。\n\n{article}",
)

write_article_chain = write_article_prompt | model | StrOutputParser()

translate_chain = (
    {"article": write_article_chain}
    | translate_prompt
    | model  #
    | StrOutputParser()  # type: ignore[var-annotated]
)


result = translate_chain.invoke({"input": "エレキギターの選び方"})
print(result)

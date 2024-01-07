from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか?",
    input_variables=["product"],
)

chain = prompt | model | StrOutputParser()
result = chain.invoke({"product": "iPhone"})

print(result)

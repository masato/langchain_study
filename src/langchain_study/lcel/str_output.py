"""LCELのPromptとStrOutputParser."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

prompt = PromptTemplate.from_template(
    """料理のレシピを考えてください。

料理名: {dish}""",
)


def upper(inp: str) -> str:
    """Convert the input string to uppercase."""
    return inp.upper()


model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


output_parser = StrOutputParser()
chain = prompt | model | output_parser | RunnableLambda(upper)

result = chain.invoke({"dish": "カレー"})
print(type(result))
print(result)

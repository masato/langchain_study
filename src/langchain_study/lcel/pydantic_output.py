"""LCELのPromptとModelを組み合わせたサンプルコードです."""

from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field


class Recipe(BaseModel):
    """Represents a recipe with ingredients and steps."""

    ingretients: list[str] = Field(..., description="材料")
    steps: list[str] = Field(..., description="手順")


output_parser = PydanticOutputParser(pydantic_object=Recipe)

prompt = PromptTemplate.from_template(
    """料理のレシピを考えてください。

{format_instructions}

料理名: {dish}""",
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
).bind(response_format={"type": "json_object"})

chain = prompt | model | output_parser
result = chain.invoke({"dish": "カレー"})

print(type(result))
print(result)

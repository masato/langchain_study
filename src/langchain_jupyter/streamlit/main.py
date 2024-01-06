"""Streamlit for the LangChain application."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic.v1 import BaseModel, Field

load_dotenv()


class Journey(BaseModel):
    """Represents a journey with belongings and route."""

    belongings: list[str] = Field(description="持ち物")
    route: list[str] = Field(description="ルート")


output_parser = PydanticOutputParser(pydantic_object=Journey)

template = """旅行のプランを考えてください。

{format_instructions}

旅行先: {destination}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["destination"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

llm = ChatOpenAI(
    model=os.environ["OPENAI_API_MODEL"],
    temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),
)

chain = prompt | llm | output_parser
journey = chain.invoke({"destination": "沖縄"})
print(journey)  # noqa: T201

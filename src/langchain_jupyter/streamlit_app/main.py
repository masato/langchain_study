"""Streamlit for the LangChain application."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
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

model = os.environ.get("OPENAI_API_MODEL")
temperature = os.environ.get("OPENAI_API_TEMPERATURE")

if model is None or temperature is None:
    # Handle the error or provide default values
    sys.exit(1)

llm = ChatOpenAI(
    model=model,
    temperature=float(temperature),
)

chain = prompt | llm | output_parser
journey = chain.invoke({"destination": "沖縄"})
print(journey)  # noqa: T201

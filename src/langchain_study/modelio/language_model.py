from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
)

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか?",
    input_variables=[
        "product",
    ],
)

result = chat.invoke(
    [
        HumanMessage(content=prompt.format(product="iPhone")),
    ],
)

print(result.content)

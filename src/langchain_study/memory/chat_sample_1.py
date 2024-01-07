from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo")

result = chat.invoke(
    [
        HumanMessage(content="茶碗蒸しを作るのに必要な食材を教えて"),
    ],
)
print(result.content)

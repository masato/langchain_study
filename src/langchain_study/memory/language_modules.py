from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI()

result = chat.invoke(
    [
        HumanMessage(content="茶碗蒸しの作り方を教"),
        AIMessage(content="ChatModelからの返答である茶碗蒸しの作り方"),
        HumanMessage(content="餃子の作り方を教えて"),
        AIMessage(content="ChatModelからの返答である餃子の作り方"),
        HumanMessage(content="チャーハンの作り方を教えて"),
    ],
)

print(result.content)

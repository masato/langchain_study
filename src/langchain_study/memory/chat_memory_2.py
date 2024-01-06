import chainlit as cl
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo")

memory = ConversationBufferMemory(return_messages=True)

chain = ConversationChain(memory=memory, llm=chat)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content="私の会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください。",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    result = chain.invoke({"input": message.content})
    await cl.Message(content=result["response"]).send()

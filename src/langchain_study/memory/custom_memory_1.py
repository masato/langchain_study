import chainlit as cl
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

memory = ConversationBufferWindowMemory(
    return_messages=True,
    k=3,
)

chain = ConversationChain(
    memory=memory,
    llm=chat,
)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content="私は会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    messages = chain.memory.load_memory_variables({})["history"]
    print(f"保存されているメッセージ数: {len(messages)}")

    for saved_message in messages:
        print(f"保存されているメッセージ: {saved_message.content}")

    result = chain.invoke({"input": message.content})
    await cl.Message(content=result["response"]).send()

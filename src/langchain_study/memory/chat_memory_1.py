import chainlit as cl
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

memory = ConversationBufferMemory(return_messages=True)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(
        content="私の会話の文脈を考慮した返答ができるチャットボットです。メッセージを入力してください。",
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    memory_message_result = memory.load_memory_variables({})
    messages = memory_message_result["history"]
    messages.append(HumanMessage(content=message.content))

    result = chat.invoke(messages)
    memory.save_context(
        {
            "input": message.content,
        },
        {
            "output": f"{result.content}",
        },
    )
    await cl.Message(content=f"{result.content}").send()

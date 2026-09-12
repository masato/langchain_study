import os
import sys

import chainlit as cl
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

redis_url = os.environ.get("REDIS_URL")

if redis_url is None:
    sys.exit(1)


@cl.on_chat_start
async def on_chat_start() -> None:
    thread_id = None
    while not thread_id:
        res = await cl.AskUserMessage(
            content="私の会話の文脈を考慮した返答ができるチャットボットです。スレッドIDを入力してください",
            timeout=600,
        ).send()
        if res:
            thread_id = res.get("output")

    history = RedisChatMessageHistory(
        session_id=thread_id,
        url=redis_url,
    )

    memory = ConversationBufferMemory(
        return_messages=True,
        chat_memory=history,
    )

    chain = ConversationChain(
        memory=memory,
        llm=chat,
    )

    memory_message_result = memory.load_memory_variables({})

    messages = memory_message_result["history"]

    for message in messages:
        if isinstance(message, HumanMessage):
            await cl.Message(
                author="User",
                content=f"{message.content}",
            ).send()
        else:
            await cl.Message(
                author="Chatbot",
                content=f"{message.content}",
            ).send()

    cl.user_session.set("chain", chain)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    chain = cl.user_session.get("chain")

    if chain is not None:
        result = chain.invoke(input={"input": message.content})
        await cl.Message(content=result["response"]).send()

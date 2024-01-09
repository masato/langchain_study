import chainlit as cl


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content="準備ができました。メッセージを入力してください").send()


@cl.on_message
async def on_message(input_message: cl.Message) -> None:
    print(f"あなたのメッセージ: {input_message}")
    await cl.Message(content="こんにちは").send()

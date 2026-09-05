from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI


class LogCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(
        self: "LogCallbackHandler",
        serialized: dict[str, Any],  # noqa: ARG002
        messages: list[list[BaseMessage]],
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        print("ChatModelの実行を開始します...")
        print(f"入力: {messages}")

    def on_chain_start(
        self: "LogCallbackHandler",
        serialized: dict[str, Any],  # noqa: ARG002
        inputs: dict[str, Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        print("Chainの実行を開始します...")
        print(f"入力: {inputs}")


chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",
    callbacks=[
        LogCallbackHandler(),
    ],
)

result = chat.invoke(
    [
        HumanMessage(content="こんにちは!"),
    ],
)

print(result.content)

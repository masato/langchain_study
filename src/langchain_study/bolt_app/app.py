"""Bolt app for Langchain."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import MomentoChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from slack_bolt import Ack, App, BoltContext, Say
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.blocks import (
    ContextBlock,
    DividerBlock,
    MarkdownTextObject,
    SectionBlock,
)

if TYPE_CHECKING:
    from langchain_core.outputs.llm_result import LLMResult


CHAT_UPDATE_INTERVAL_SEC = 1


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

app = App(
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
    token=os.environ["SLACK_BOT_TOKEN"],
    process_before_response=True,
)


class SlackStreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for Slack streaming."""

    last_send_time = time.time()
    message = ""

    def __init__(self: SlackStreamingCallbackHandler, channel: str, ts: str) -> None:
        """Initialize the SlackStreamingCallbackHandler.

        Parameters
        ----------
        channel : str
            The channel to send updates to.
        ts : str
            The timestamp of the message to update.

        Returns
        -------
        None
            This is a constructor method and does not return anything.
        """
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        # 投稿を更新した累計回数カウンタ
        self.update_count = 0

    def on_llm_new_token(
        self: SlackStreamingCallbackHandler,
        token: str,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Handle the new token received during the LLM process.

        Parameters
        ----------
        token : str
            The new token received.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        None
            Nothing to return.
        """
        self.message += token

        now = time.time()
        if now - self.last_send_time > self.interval:
            app.client.chat_update(
                channel=self.channel,
                ts=self.ts,
                text=f"{self.message}...",
            )

            self.last_send_time = now
            self.update_count += 1

            # update_count が現在の更新間隔 x 10 より多くなるたびに更新間隔を 2 倍にする
            if self.update_count / 10 > self.interval:
                self.interval = self.interval * 2

    def on_llm_end(
        self: SlackStreamingCallbackHandler,
        response: LLMResult,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Handle the end of the LLM process.

        Parameters
        ----------
        response : LLMResult
            The response from the LLM process.
        kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        None
            Nothing to return.
        """
        message_context = (
            "OpenAI APIで生成される情報は不正確または不適切な場合がありますが、"
            "当社の見解を述べるものではありません。"
        )
        message_blocks = [
            SectionBlock(text=MarkdownTextObject(text=self.message)),
            DividerBlock(),
            ContextBlock(elements=[MarkdownTextObject(text=message_context)]),
        ]

        app.client.chat_update(
            channel=self.channel,
            ts=self.ts,
            text=self.message,
            blocks=message_blocks,
        )


# @app.event("app_mention")
def handle_mention(event: dict, say: Say) -> None:
    """Handle mention event.

    Parameters
    ----------
    event : dict
        The mention event data.
    say : Say
        The say function to send a message.

    Returns
    -------
    None
        This function does not return anything.
    """
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub(r"<@.*>", "", event["text"])

    # 投稿のキー(=Momentoキー):初回=event["ts"],2回目以降=event["thread_ts"]
    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = str(result["ts"])

    history = MomentoChatMessageHistory.from_client_params(
        session_id=id_ts,
        cache_name=os.environ["MOMENTO_CACHE"],
        ttl=timedelta(hours=float(os.environ["MOMENTO_TTL"])),
    )

    messages: list[BaseMessage] = []
    messages.append(SystemMessage(content="You are a good assistant."))
    messages.extend(history.messages)
    messages.append(HumanMessage(content=message))

    history.add_user_message(message)

    callback = SlackStreamingCallbackHandler(channel=channel, ts=ts)

    llm = ChatOpenAI(
        model=os.environ["OPENAI_API_MODEL"],
        temperature=float(os.environ["OPENAI_API_TEMPERATURE"]),
        streaming=True,
        callbacks=[callback],
    )

    ai_message = llm(messages)
    history.add_message(ai_message)


def just_ack(ack: Ack) -> None:
    """Acknowledge the event.

    Parameters
    ----------
    ack : Ack
        The ack function.

    Returns
    -------
    None
        This function does not return anything.
    """
    ack()


app.event("app_mention")(ack=just_ack, lazy=[handle_mention])


def handler(event: dict, context: BoltContext) -> dict[str, Any]:
    """Handle the event and return a dictionary.

    Parameters
    ----------
    event : dict
        The event data.
    context : BoltContext
        The Bolt context.

    Returns
    -------
    dict[str, Any]
        The dictionary containing the response.
    """
    logger.info("handler called")
    header = event["headers"]
    logger.info(json.dumps(header))

    if "x-slack-retry-num" in header:
        logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
        return {"statusCode": 200, "body": json.dumps({"message": "ok"})}

    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)


if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()

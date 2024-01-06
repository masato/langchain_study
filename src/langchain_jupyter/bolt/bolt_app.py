import json
import logging
import os
import re
import time
from datetime import timedelta
from typing import Any

from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.memory import MomentoChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, LLMResult, SystemMessage
from slack_bolt import App, BoltContext, Say
from slack_bolt.adapter.aws_lambda import SlackRequestHandler
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.models.blocks import (
    ContextBlock,
    DividerBlock,
    MarkdownTextObject,
    SectionBlock,
)

CHAT_UPDATE_INTERVAL_SEC = 1

load_dotenv()

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
    last_send_time = time.time()
    message = ""

    def __init__(self, channel: str, ts: str) -> None:
        self.channel = channel
        self.ts = ts
        self.interval = CHAT_UPDATE_INTERVAL_SEC
        # 投稿を更新した累計回数カウンタ
        self.update_count = 0

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
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

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        message_context = "OpenAI APIで生成される情報は不正確または不適切な場合がありますが、当社の見解を述べるものではありません。"
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
    channel = event["channel"]
    thread_ts = event["ts"]
    message = re.sub(r"<@.*>", "", event["text"])

    # 投稿のキー(=Momentoキー)：初回=event["ts"],2回目以降=event["thread_ts"]
    id_ts = event["ts"]
    if "thread_ts" in event:
        id_ts = event["thread_ts"]

    result = say("\n\nTyping...", thread_ts=thread_ts)
    ts = result["ts"]

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


def just_ack(ack: Any) -> None:
    ack()


app.event("app_mention")(ack=just_ack, lazy=[handle_mention])

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()


def handler(event: dict, context: BoltContext) -> dict[str, Any]:
    logger.info("handler called")
    header = event["headers"]
    logger.info(json.dumps(header))

    if "x-slack-retry-num" in header:
        logger.info("SKIP > x-slack-retry-num: %s", header["x-slack-retry-num"])
        return {"statusCode": 200, "body": json.dumps({"message": "ok"})}

    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)

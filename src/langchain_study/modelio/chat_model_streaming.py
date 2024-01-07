from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    streaming=True,
    callbacks=[
        StreamingStdOutCallbackHandler(),
    ],
)

resp = chat.invoke(
    [
        HumanMessage(content="おいしいステーキの焼き方を教えて"),
    ],
)
print(resp.content)

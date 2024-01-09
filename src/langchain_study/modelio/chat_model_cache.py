import time

from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

set_llm_cache(InMemoryCache())
chat = ChatOpenAI()

start = time.time()
result = chat.invoke(
    [
        HumanMessage(content="こんにちは!"),
    ],
)
end = time.time()
print(result.content)
print(f"処理時間: {end - start}秒")

start = time.time()
result = chat.invoke(
    [
        HumanMessage(content="こんにちは!"),
    ],
)
end = time.time()
print(result.content)
print(f"処理時間: {end - start}秒")

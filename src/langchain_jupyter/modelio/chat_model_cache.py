import time

from langchain.cache import InMemoryCache
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain.schema import HumanMessage

set_llm_cache(InMemoryCache())
chat = ChatOpenAI()

start = time.time()
result = chat(
    [
        HumanMessage(content="こんにちは!"),
    ],
)
end = time.time()
print(result.content)
print(f"処理時間: {end - start}秒")

start = time.time()
result = chat(
    [
        HumanMessage(content="こんにちは!"),
    ],
)
end = time.time()
print(result.content)
print(f"処理時間: {end - start}秒")

import random

from langchain.agents import AgentType, Tool, initialize_agent
from langchain_community.tools import WriteFileTool
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",
)

tools: list[BaseTool] = []
tools.append(
    WriteFileTool(
        root_dir="./",
    ),
)


def min_limit_random_number(min_limit: str) -> int:
    return random.randint(int(min_limit), 100000)  # noqa: S311


tools.append(
    Tool(
        name="Random",
        func=min_limit_random_number,
        description="特定の最小値以上のランダムな数字を生成することができます。",
    ),
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.run(
    "10以上のランダムな数字を生成してrandom.txtというファイルに保存してください。",
)

print(f"実行結果: {result}")

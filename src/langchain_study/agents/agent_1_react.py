from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    load_tools,
)
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
)


tools = load_tools(["requests_get"])
prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm=chat,
    prompt=prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,  # type: ignore  # noqa: PGH003
    tools=tools,
    verbose=True,
)

result = agent_executor.invoke(
    {
        "input": """以下のURLにアクセスして東京の天気を調べて日本語で答えてください。
https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json
""",
    },
)

print(f"result: {result}")

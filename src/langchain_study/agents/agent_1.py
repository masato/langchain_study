from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
)

tools = load_tools(
    [
        "requests",
    ],
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.run(
    """以下のURLにアクセスして東京の天気を調べて日本語で答えてください。
https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json
""",
)

print(f"result: {result}")

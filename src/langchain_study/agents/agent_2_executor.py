from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    load_tools,
)
from langchain_community.tools import WriteFileTool
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",
)

tools = load_tools(
    ["requests_get", "serpapi"],
    llm=chat,
)
tools.append(
    WriteFileTool(
        root_dir="./",
    ),
)
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
        "input": "北海道の名産品を調べて日本語でresult.txtというファイルに保存してください。",
    },
)

print(f"result: {result}")

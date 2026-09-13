from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_community.tools.file_management import WriteFileTool
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",
)

tools = load_tools(
    ["serpapi"],
    llm=chat,
)

tools.append(
    WriteFileTool(
        root_dir="./",
    ),
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.run(
    "北海道の名産品を調べて日本語でresult.txtというファイルに保存してください。",
)

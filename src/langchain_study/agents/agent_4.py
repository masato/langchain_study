from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.retrievers import WikipediaRetriever
from langchain_community.tools.file_management import WriteFileTool
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
)

tools: list[BaseTool] = []
tools.append(
    WriteFileTool(
        root_dir="./",
    ),
)

retriever = WikipediaRetriever(
    lang="ja",
    doc_content_chars_max=500,
    top_k_results=2,
    wiki_client=None,
)

tools.append(
    create_retriever_tool(
        name="WikipediaRetriever",
        description="受け取った単語に関するWikipedia記事を取得することができます。",
        retriever=retriever,
    ),
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.run(
    "スコッチウイスキーについてWikipediaで調べて概要を日本語でresult.txtというファイルに保存してください。",
)

print(f"実行結果: {result}")

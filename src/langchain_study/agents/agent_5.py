from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",
)

tools: list[BaseTool] = []

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

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
)

result = agent.run(
    "スコッチウイスキーについてWikipediaで調べて概要を日本語で概要をまとめてください。",
)
print(f"1 回目の実行結果: {result}")

result_2 = agent.run("以前の指示をもう一度実行してください。")
print(f"2 回目の実行結果: {result_2}")

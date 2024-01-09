import chainlit as cl
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
    load_tools,
)
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo-1106",
)

tools = load_tools(
    ["requests_get", "serpapi"],
    llm=chat,
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
    max_iterations=3,
    handle_parsing_errors=True,
)


@cl.on_chat_start
async def on_chat_start() -> None:
    await cl.Message(content="Agentの初期化が完了しました。").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    result = agent_executor.invoke(
        input={"input": message.content},
        config={"callbacks": [cl.LangchainCallbackHandler()]},
    )
    print(result)
    await cl.Message(content=result["output"]).send()

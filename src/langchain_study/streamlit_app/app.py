"""Streamlit app for Langchain."""
from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import streamlit as st
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI

if TYPE_CHECKING:
    from langchain.agents.agent import AgentExecutor


def create_agent_chain() -> AgentExecutor:
    """Create and initialize an agent chain for Langchain.

    Returns
    -------
        AgentExecutor: The initialized agent chain.
    """
    model = os.environ.get("OPENAI_API_MODEL")
    temperature = os.environ.get("OPENAI_API_TEMPERATURE")

    if model is None or temperature is None:
        sys.exit(1)

    chat = ChatOpenAI(
        model=model,
        temperature=float(temperature),
        streaming=True,
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    memory = ConversationBufferMemory(memory_key="memory", return_messages=True)

    tools = load_tools(["ddg-search", "wikipedia"])
    return initialize_agent(
        tools,
        chat,
        agent=AgentType.OPENAI_FUNCTIONS,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )


st.title("langchain-streamlit-app")

if "agent_chain" not in st.session_state:
    st.session_state.agent_chain = create_agent_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())
        agent_chain = create_agent_chain()
        response = st.session_state.agent_chain.run(prompt, callbacks=[callback])

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# app.py - Streamlit + LangChain 예제 (중복 출력 해결 및 대화 히스토리 유지)
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Agent 관련 모듈
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory

# 📌 환경 변수 로드
load_dotenv()

# 📌 대화 히스토리를 Session State에 유지
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 히스토리 리스트 초기화

# 📌 Agent 생성 함수
def create_agent_chain():
    chat = ChatOpenAI(
        model_name=os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo"),
        temperature=float(os.getenv("OPENAI_API_TEMPERATURE", 0.5)),
        max_tokens=500
    )

    # 🔧 도구 로드
    tools = load_tools(["ddg-search", "wikipedia"])

    # 🔧 프롬프트 로드
    prompt = hub.pull("hwchase17/openai-tools-agent")

    # 📝 ConversationBufferMemory 초기화
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # 🛠️ Agent 생성
    agent = create_openai_tools_agent(chat, tools, prompt)

    # 🚀 Agent Executor 생성
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True
    )

# 📌 Streamlit 제목 및 설명
st.title("🚀 Dani's Chatbot")
st.write("AWS EC2 + LangChain Agents를 활용한 Streamlit 챗봇입니다. 🎉")

# 💬 대화 히스토리 출력 (세션 기반)
for message in st.session_state.messages:
    if message["type"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["type"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# 🟡 사용자 입력 처리
prompt = st.chat_input("What's up?")

if prompt:
    # 🗨️ 사용자 메시지 즉시 화면 출력
    with st.chat_message("user"):
        st.markdown(prompt)

    # 💾 세션 히스토리에 사용자 메시지 추가
    st.session_state.messages.append({"type": "user", "content": prompt})

    # 🤖 AI 응답 생성 및 즉시 출력
    with st.chat_message("assistant"):
        agent_chain = create_agent_chain()

        try:
            response = agent_chain.invoke({"input": prompt})
            output = response.get("output", "No response generated.")

            # 💾 세션 히스토리에 AI 메시지 추가
            st.session_state.messages.append({"type": "assistant", "content": output})

            # AI 응답 즉시 출력
            st.markdown(output)

        except Exception as e:
            st.error(f"오류 발생: {e}")
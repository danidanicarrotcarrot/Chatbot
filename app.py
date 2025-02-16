# app.py - Streamlit + LangChain 예제 with Chat History 표시 (수정됨)
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage

# Agent 관련 모듈
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_community.callbacks import StreamlitCallbackHandler

# Memory 관련 모듈
from langchain.memory import ConversationBufferMemory

# 📌 환경 변수 로드
load_dotenv()

# 📌 대화 히스토리를 Session State에 저장
if "chat_history" not in st.session_state:
    st.session_state.chat_history = StreamlitChatMessageHistory()

# 📌 Agent 생성 함수
def create_agent_chain(history):
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
        chat_memory=history,
        memory_key='chat_history',
        return_messages=True,
        output_key=None
    )

    # 🛠️ Agent 생성
    agent = create_openai_tools_agent(chat, tools, prompt)

    # 🚀 Agent Executor 생성
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=False
    )

# 📌 Streamlit 제목 및 설명
st.title("🚀 AWS EC2 + LangChain Agent Chatbot")
st.write("LangChain Agents를 활용한 Streamlit 챗봇입니다. 🎉")

# 💬 이전 대화 히스토리 출력
st.subheader("💬 이전 대화 히스토리")
for message in st.session_state.chat_history.messages[:-1]:  # 마지막 메시지 제외
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            # st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            # st.markdown(message.content)

# 🟡 사용자 입력 처리
prompt = st.chat_input("What's up?")

if prompt:
    # 🗨️ 사용자 메시지 즉시 표시
    with st.chat_message("user"):
        st.markdown(prompt)

    # 🤖 AI 응답 생성 및 출력
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())  # 콜백 핸들러 추가
        agent_chain = create_agent_chain(st.session_state.chat_history)

        try:
            response = agent_chain.invoke({"input": prompt})
            output = response.get("output", "No response generated.")

            # 📝 Session State 히스토리 업데이트
            st.session_state.chat_history.add_user_message(prompt)
            st.session_state.chat_history.add_ai_message(output)

            # AI 응답 즉시 표시
            st.markdown(output)

        except Exception as e:
            st.error(f"오류 발생: {e}")

# 🔄 📝 전체 대화 히스토리 즉시 업데이트
st.divider()
st.subheader("📝 전체 대화 내역")
for message in st.session_state.chat_history.messages:
    if isinstance(message, HumanMessage):
        st.markdown(f"👤 **사용자:** {message.content}")
    elif isinstance(message, AIMessage):
        st.markdown(f"🤖 **AI:** {message.content}")
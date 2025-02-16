# app.py - Streamlit + LangChain 예제 with Agent
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# Agent 관련 모듈
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, load_tools
from langchain_community.callbacks import StreamlitCallbackHandler

# 📌 환경 변수 로드
load_dotenv()

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
    
    # 🛠️ Agent 생성
    agent = create_openai_tools_agent(chat, tools, prompt)
    
    # 🚀 Agent Executor 생성
    return AgentExecutor.from_agent_and_tools(
        agent=agent, 
        tools=tools,
        verbose=True
    )

# 📌 Streamlit 제목 및 설명
st.title("🚀 AWS EC2 + LangChain Agent Chatbot")
st.write("LangChain Agents를 활용한 Streamlit 챗봇입니다. 🎉")

# 📌 Chat History 초기화
history = StreamlitChatMessageHistory()

# 🔁 이전 메시지 표시
for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# 🟡 사용자 입력 처리
prompt = st.chat_input("What's up?")

if prompt:
    # 🗨️ 사용자 메시지 출력
    with st.chat_message("user"):
        history.add_user_message(prompt)
        st.markdown(prompt)

    # 🤖 AI 응답 출력
    with st.chat_message("assistant"):
        callback = StreamlitCallbackHandler(st.container())  # 콜백 핸들러 추가
        agent_chain = create_agent_chain()

        try:
            response = agent_chain.invoke(
                {"input": prompt, "callbacks": [callback]}
            )
            output = response.get("output", "No response generated.")
            history.add_ai_message(output)
            st.markdown(output)
        except Exception as e:
            st.error(f"오류 발생: {e}")
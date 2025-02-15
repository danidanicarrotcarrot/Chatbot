# app.py - Streamlit + LangChain 예제
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 제목 및 설명
st.title("🚀 Hello, AWS EC2 from Streamlit with LangChain!")
st.write("AWS EC2에서 Streamlit과 LangChain을 이용한 AI 챗봇입니다. 🎉")

# Chat History 초기화
history = StreamlitChatMessageHistory()

# 이전 메시지 출력
for message in history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat Input & Chat Message 사용
prompt = st.chat_input("What's up?")

if prompt:
    # 사용자 메시지 출력
    with st.chat_message("user"):
        history.add_user_message(prompt)
        st.markdown(prompt)

    # AI 응답 출력
    with st.chat_message("assistant"):
        response = "안녕하세요! 😊 반갑습니다!"
        history.add_ai_message(response)
        st.markdown(response)

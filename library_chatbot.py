# 💬 MoodBite - 기분 분석 및 음식 추천 챗봇
import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

# --- 1. Gemini API 키 설정 ---
try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# --- 2. LLM 및 프롬프트 설정 (캐시) ---
@st.cache_resource(show_spinner="🤖 챗봇 모델 로딩 중...")
def get_chat_chain(selected_model):
    """
    LLM, 프롬프트, 출력 파서를 결합한 기본 체인을 생성합니다.
    """
    
    # LLM 로드
    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 API 키가 유효한지, 모델 이름이 올바른지 확인해보세요.")
        st.stop()

    # 대화 프롬프트 설정
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 사용자의 기분을 분석하여 맞춤형 음식을 추천해주는 'MoodBite' 챗봇입니다. 즐거운 기분에는 상큼한 디저트를, 지친 기분에는 든든한 한 끼를 제안합니다. 항상 친절하고 유머러스하게 대답하세요. 🤖"),
            MessagesPlaceholder("history"), # 대화 기록이 들어갈 자리
            ("human", "{input}"),         # 사용자의 현재 입력이 들어갈 자리
        ]
    )
    
    # LLM 체인 생성 (프롬프트 | 모델 | 출력파서)
    # StrOutputParser()는 LLM의 출력(AIMessage)을 간단한 문자열(string)로 변환합니다.
    chain = prompt | llm | StrOutputParser()
    
    return chain

# --- 3. Streamlit UI 설정 ---

st.header("MoodBite - 기분 분석 및 음식 추천 챗봇 💬")
st.info("사용자의 대화를 분석해 기분에 맞는 음식을 추천해주는 스마트 챗봇입니다. 즐거운 기분에는 상큼한 디저트를, 지친 기분에는 든든한 한 끼를 제안합니다.")

# 채팅 기록을 Streamlit의 세션 상태(session_state)에 저장
# key="chat_messages"는 이 채팅 기록을 식별하는 고유 키입니다.
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 모델 선택
option = st.selectbox("Select Gemini Model",
    ("gemini-2.5-flash", "gemini-pro", "gemini-2.0-flash-exp"),
    index=0,
    help="가장 빠르고 효율적인 2.5 Flash 모델을 추천합니다."
)

# 선택된 모델로 LLM 체인 가져오기
simple_chain = get_chat_chain(option)

# 대화 기록을 관리하는 Runnable 생성
conversational_chain = RunnableWithMessageHistory(
    simple_chain,
    lambda session_id: chat_history, # session_id에 관계없이 항상 chat_history 사용
    input_messages_key="input",      # 프롬프트의 "{input}"에 사용자 입력을 매핑
    history_messages_key="history",  # 프롬프트의 "history"에 대화 기록을 매핑
)

# --- 4. 채팅 UI 로직 ---

# 첫 방문 시 환영 메시지 추가
if not chat_history.messages:
    chat_history.add_ai_message("안녕하세요! 만나서 반가워요. 😊 기분을 알려주세요! 그에 맞는 음식을 추천해드릴게요.")

# 이전 대화 기록 모두 출력
for msg in chat_history.messages:
    st.chat_message(msg.type).write(msg.content)

# 사용자 입력 받기
col1, col2 = st.columns([6, 2])

with col1:
    if prompt_message := st.chat_input("기분을 알려주세요!"):
        st.chat_message("user").write(prompt_message)

with col2:
    if st.button("메뉴 정해주기"):
        st.chat_message("assistant").write("기분에 맞는 음식을 추천해드릴게요! 잠시만 기다려주세요. 🍴")

# AI 응답 생성 및 출력
if prompt_message:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # config: session_id는 아무 값이나 넣어도 chat_history를 사용하도록 설정됨
            config = {"configurable": {"session_id": "any_id"}}
            
            # 체인 실행 (RAG와 달리, 'context'가 없는 간단한 문자열을 반환)
            response = conversational_chain.invoke(
                {"input": prompt_message},
                config
            )
            st.write(response)

import os
import streamlit as st
import nest_asyncio

# Streamlit에서 비동기 작업을 위한 이벤트 루프 설정
nest_asyncio.apply()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser # 사용되지는 않지만, LangChain 템플릿에 포함되어 있어 유지합니다.
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory

# pysqlite3를 사용하여 ChromaDB의 호환성 문제 해결
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_chroma import Chroma


# Gemini API 키 설정
try:
    # 환경 변수 대신 st.secrets를 사용하여 Streamlit Secrets에서 API 키 로드
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception as e:
    st.error("⚠️ GOOGLE_API_KEY를 Streamlit Secrets에 설정해주세요!")
    st.stop()

# cache_resource로 한번 실행한 결과 캐싱해두기 (PDF 로드 및 분할)
@st.cache_resource
def load_and_split_pdf(file_path):
    # PDF 파일이 존재하는지 확인
    if not os.path.exists(file_path):
        st.error(f"⚠️ 파일 경로 오류: '{file_path}'를 찾을 수 없습니다. 파일을 프로젝트 폴더에 넣어주세요.")
        st.stop()

    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    st.info(f"📄 {len(split_docs)}개의 텍스트 청크로 분할했습니다.")

    persist_directory = "./chroma_db"
    st.info("🤖 임베딩 모델 로드 중... (첫 실행 시 모델 다운로드)")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.info("🔢 벡터 임베딩 생성 및 저장 중...")
    vectorstore = Chroma.from_documents(
        split_docs,
        embeddings,
        persist_directory=persist_directory
    )
    # Chroma DB를 디스크에 저장 (캐시가 아닌 실제 파일로)
    vectorstore.persist()
    st.success("💾 벡터 데이터베이스 생성 완료!")
    return vectorstore

# 만약 기존에 저장해둔 ChromaDB가 있는 경우, 이를 로드
@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 임베딩 함수는 동일해야 로드 가능
    if os.path.exists(persist_directory) and os.path.exists(os.path.join(persist_directory, "chroma-collections.parquet")):
        st.info("🔄 기존 벡터 데이터베이스 로드 중...")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        st.info("✨ 새로운 벡터 데이터베이스 생성 시작...")
        return create_vector_store(_docs)
    
# PDF 문서 로드-벡터 DB 저장-검색기-히스토리 모두 합친 Chain 구축
@st.cache_resource
def initialize_components(selected_model):
    file_path = "[챗봇프로그램및실습] 부경대학교 규정집.pdf"
    
    # 1. 문서 로드 및 분할
    pages = load_and_split_pdf(file_path)
    
    # 2. 벡터 저장소 (Chroma DB) 로드 또는 생성
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # 3. 채팅 히스토리 요약 시스템 프롬프트 (Contextualization)
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. \
    Your output should be a standalone question only."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 4. 질문-답변 시스템 프롬프트 (QA)
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Your name is '부경대 규정 봇' (Pukyong National University Regulation Bot).
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Keep the answer perfect. please use imogi with the answer.
    대답은 한국어로 하고, 존댓말을 써줘. 답변에 필요한 근거는 반드시 'retrieved context'에서 찾아야 합니다.\

    Context:
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )

    # 5. LLM 및 체인 설정
    try:
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.3, # 답변의 일관성을 위해 온도를 낮춤
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"❌ Gemini 모델 '{selected_model}' 로드 실패: {str(e)}")
        st.info("💡 'gemini-2.5-flash' 또는 'gemini-pro' 모델을 사용해보세요.")
        raise
        
    # 히스토리 인식 검색기 체인
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # 문서 스터핑 및 답변 생성 체인
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # 전체 RAG 체인
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# Streamlit UI
st.header("MoodBite 🤖🍽️")
st.markdown("### 사용자 대화를 분석해 기분을 짐작하고 그에 맞는 음식을 추천해주는 스마트 챗봇. 즐거운 기분에는 상큼한 디저트를, 지친 기분에는 든든한 한 끼를 제안합니다. 😊")

# Gemini 모델 선택
option = st.selectbox("🤖 사용할 Gemini 모델을 선택해주세요:",
    ("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash-exp"),
    index=0,
    help="Gemini 2.5 Flash가 가장 빠르고 효율적이며 비용 효율성이 높습니다."
)

# 챗봇 초기화 및 로드
try:
    with st.spinner("🔧 챗봇 초기화 중... 잠시만 기다려주세요 (문서 처리 및 LLM 로드)"):
        rag_chain = initialize_components(option)
    st.success("✅ 챗봇이 준비되었습니다!")
except Exception as e:
    # initialize_components 내에서 이미 오류 메시지가 출력되었으므로, 여기서는 제어만 함
    st.stop()

# 세션 히스토리 설정
chat_history = StreamlitChatMessageHistory(key="chat_messages")

# 채팅 히스토리와 RAG 체인을 결합한 최종 Conversational Chain
conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id: chat_history,
    input_messages_key="input",
    history_messages_key="history",
    output_messages_key="answer",
)

# 초기 메시지 설정 및 기존 메시지 표시
if not chat_history.messages:
    chat_history.messages.append({"role": "assistant", 
                                 "content": "안녕하세요! MoodBite입니다. 😊 기분을 말해주시면 그에 맞는 음식을 추천해드릴게요!"})
                                 
for msg in chat_history.messages:
    # LangChain의 message.type을 Streamlit의 role로 변환
    role = "assistant" if msg.type == "ai" else "user"
    st.chat_message(role).write(msg.content)


# 사용자 입력 처리
col1, col2 = st.columns([6, 2])

with col1:
    if prompt_message := st.chat_input("기분을 알려주세요!"):
        st.chat_message("user").write(prompt_message)

with col2:
    if st.button("메뉴 정해주기"):
        st.chat_message("assistant").write("음... 기분에 맞는 메뉴를 추천해드릴게요! 잠시만 기다려주세요. 🍴")

    with st.chat_message("assistant"):
        with st.spinner("답변을 생성 중입니다..."):
            config = {"configurable": {"session_id": "any"}}
            
            # RunnableWithMessageHistory 호출
            response = conversational_rag_chain.invoke(
                {"input": prompt_message},
                config
            )
            
            answer = response.get('answer', "죄송합니다. 답변을 생성하는 데 문제가 발생했습니다. 😥")
            st.write(answer)
            
            # 참고 문서 표시
            context_docs = response.get('context', [])
            if context_docs:
                with st.expander("🔍 답변에 사용된 참고 문서 (클릭해서 내용 확인)"):
                    for i, doc in enumerate(context_docs):
                        source_info = f"**출처:** {doc.metadata.get('source', '알 수 없음')} (페이지: {doc.metadata.get('page', '알 수 없음')})"
                        st.markdown(f"---")
                        st.markdown(source_info)
                        st.markdown(f"**내용 요약:** {doc.page_content[:200]}...") # 긴 내용은 일부만 보여줌
            else:
                st.info("문서에서 관련된 정보를 찾지 못했습니다. 질문을 구체화해주세요.")

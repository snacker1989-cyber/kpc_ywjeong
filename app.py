import streamlit as st

from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from RAGbuilder import get_embeddings, VSTORE_DIR

st.set_page_config(page_title="한국생산성본부 내부규정 도우미", page_icon="🏢", layout="wide")

# --- 캐시를 이용한 리소스 초기화 ---
@st.cache_resource
def get_retriever():
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    return db

@st.cache_resource
def get_qa_chain():
    db = get_retriever()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})   # vs. search_type="mmr"
    
    # 스트리밍 지원을 위해 llm 설정
    llm = ChatOllama(model="gemma3", temperature=0.2)      # vs. llama3.1 - 자원이 여유가 될 때
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "# 역할 및 목표 "
                   "당신은 한국생산성본부의 사규, 규정, 규칙, 가이드라인 등 내부 지침을 숙지한 AI 어시스턴트입니다. 당신의 주요 임무는 "
                   "1. 사용자가 특정 상황과 함께 해결하고 싶은 문제나 고민을 제시하면 이에 적용될 수 있는 규정의 이름과 조항 번호 등을 제시합니다. "
                   "2. 사용자가 특정 상황과 규정을 제시하며 특정 상황을 해당 규정으로 해결할 수 있는지 질문하면, 문헌적 해석을 우선하여 답변하되, 문헌적 해석이 불가능한 경우에 한하여 대한민국 관련 법령 및 합리적 추론을 활용하여 답변합니다. "
                   "3. 모든 답변은 {context}를 토대로 제시하고, 답변에 참고가 된 내용이 무엇인지 근거가 되는 조항(예: 인사규정 제1조제2항, 복무규정 제3조제4항제5호 등)을 반드시 밝혀주세요. "
                   "# 주의사항 "
                   "답변에 근거가 없거나 그 결과물이 확실하지 않다면, 없는 내용을 지어내는 것보다 모른다고 답하는 것이 더 낫습니다. "
                   "근거가 되는 규정들이 상충되거나 해석의 여지가 있다면, 그 점을 밝히고 담당부서에 문의하라고 안내해주세요. "),
        ("human", "{question}")
    ])
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- 사이드바 구성 --- #### 테스트 할 때는 죽여놓자... 무겁다
with st.sidebar:
    st.title("📂 문서 관리 현황")
    st.info("현재 벡터 저장소에 저장된 정보입니다.")
    
    db = get_retriever()
    all_docs = db.get()
    
    if all_docs and all_docs['metadatas']:
        sources = {}
        for m in all_docs['metadatas']:
            source = Path(m.get('source', 'Unknown')).name
            sources[source] = sources.get(source, 0) + 1
        
        st.metric("총 청크(Chunk) 수", f"{len(all_docs['documents'])} 개")
        
        st.markdown("### 📄 학습된 파일 목록")
        for name, count in sources.items():
            st.write(f"- {name} ({count} chunks)")
    else:
        st.warning("저장된 문서가 없습니다. 'build'를 먼저 실행해주세요.")

    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

        
# --- 메인 화면 구성 ---
st.title("🏢 한국생산성본부 내부규정 도우미")
st.caption("한국생산성본부 임직원을 위한 규정 안내 챗봇입니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 기록 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력 및 처리
if prompt := st.chat_input("내부규정에 대하여 궁금하신 내용을 입력하세요 (예: 쓸 수 있는 휴가들의 종류를 알려줘)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.empty() # 스트리밍 출력을 위한 빈 공간
        full_response = ""
        
        with st.spinner("이런저런 규정들을 살펴보는 중..."):
            qa_chain = get_qa_chain()
            # stream 메소드를 사용하여 한 글자씩 출력
            for chunk in qa_chain.stream(prompt):
                full_response += chunk
                container.markdown(full_response + "▌")
            container.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})
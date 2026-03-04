import chainlit as cl

from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from RAGbuilder import get_embeddings, VSTORE_DIR 

# --- 리소스 초기화 함수 ---
def get_retriever():
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    return db

def create_qa_chain():
    db = get_retriever()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # 로컬 LLM 설정 (gemma3)
    llm = ChatOllama(model="gemma3", temperature=0.2)
    
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
    
    # Chain 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# --- Chainlit 이벤트 핸들러 ---

@cl.on_chat_start
async def start():
    """채팅 세션이 시작될 때 실행되는 함수"""
    
    # 1. 사이드바에 현재 학습된 문서 정보 표시 (선택 사항)
    db = get_retriever()
    all_docs = db.get()
    
    if all_docs and all_docs['metadatas']:
        sources = {}
        for m in all_docs['metadatas']:
            source = Path(m.get('source', 'Unknown')).name
            sources[source] = sources.get(source, 0) + 1
        
        status_msg = f"현재 {len(all_docs['documents'])}개의 규정 조각이 학습되어 있습니다.\n\n"
        status_msg += "\n".join([f"- {name} ({count} chunks)" for name, count in sources.items()])
        
        # 안내 메시지 전송
        await cl.Message(content=f"🏢 **한국생산성본부 내부규정 도우미**가 준비되었습니다.\n\n{status_msg}").send()
    else:
        await cl.Message(content="⚠️ 학습된 문서가 없습니다. 벡터DB를 먼저 확인해주세요.").send()

    # 2. QA Chain을 세션에 저장 (사용자별로 독립적인 체인 유지)
    chain = create_qa_chain()
    cl.user_session.set("qa_chain", chain)


@cl.on_message
async def main(message: cl.Message):
    """사용자가 메시지를 보낼 때마다 실행되는 함수"""
    
    # 세션에서 체인 가져오기
    chain = cl.user_session.get("qa_chain")
    
    # 답변을 위한 빈 메시지 생성 (스트리밍용)
    msg = cl.Message(content="")
    
    # LangChain의 stream 기능을 활용하여 실시간 답변 출력
    # Chainlit의 'astream' 혹은 'stream'을 통해 토큰 단위로 전송합니다.
    async for chunk in chain.astream(
        message.content,
        config=cl.ChatSettings.load().to_dict() # 필요한 경우 설정 로드
    ):
        await msg.stream_token(chunk)
    
    # 최종 메시지 전송
    await msg.send()
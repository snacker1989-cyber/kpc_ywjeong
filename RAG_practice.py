import os
import sys

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
# from langchain_classic.chains import RetrievalQA

DOCS_DIR = "pdfs"
VSTORE_DIR = "vectorstore"


def load_and_split(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    return splitter.split_documents(docs)

def get_embeddings():
    #emb = OllamaEmbeddings(model="embeddinggemma")
    emb = OllamaEmbeddings(model="qwen3-embedding")
    return emb

def build_vectorstore(batch_size: int = 100):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)

    total_indexed = 0
    pdfs = list(Path(DOCS_DIR).glob("*.pdf"))

    if not pdfs:
        print("No PDF files found in:", DOCS_DIR)
        return
    
    for p in pdfs:
        print("Processing:", p)
        docs = load_and_split(p)
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            if not batch:
                continue
            db.add_documents(batch)
            total_indexed += len(batch)
            print(f"  Indexed {total_indexed} documents so far")
            # free memory immediately after persisting
            del batch
            import gc
            gc.collect()
        del docs
        import gc
        gc.collect()
    
    print("Vectorstore built and persisted at:", VSTORE_DIR)


def check_vectorstore_contents() -> None:       # 벡터 저장소에 저장된 PDF 파일 목록과 청크 수 확인
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    
    # 벡터 저장소에 저장된 문서의 수 확인
    all_docs = db.get()
    
    if not all_docs or not all_docs['documents']:
        print("Vectorstore is empty.")
        return
    
    # 메타데이터에서 source 정보 추출
    sources = {}
    for metadata in all_docs['metadatas']:
        source = metadata.get('source', 'Unknown')
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    print("\n=== Vectorstore Contents ===")
    print(f"Total embedded chunks: {len(all_docs['documents'])}")
    print("\nEmbedded PDF files:")
    for source, count in sources.items():
        pdf_name = Path(source).name
        print(f"  - {pdf_name}: {count} chunks")
    print()


def reset_vectorstore() -> None:    # 벡터 저장소 초기화
    import shutil
    vstore_path = Path(VSTORE_DIR)
    
    if not vstore_path.exists():
        print(f"Vectorstore does not exist: {VSTORE_DIR}")
        return
    
    try:
        shutil.rmtree(vstore_path)
        print(f"✓ Vectorstore has been reset: {VSTORE_DIR}")
    except Exception as e:
        print(f"✗ Vectorstore reset failed: {e}")



messages_with_questions = [
    (
        "system",
        "당신은 한국생산성본부의 사규, 규정, 규칙, 가이드라인 등 내부 지침을 숙지한 AI 어시스턴트입니다."
        " {context}를 토대로 답변하고, 답변에 참고가 된 내용이 무엇인지 출처를 반드시 밝혀주세요."
        " 답변에 근거가 없거나 그 결과물이 확실하지 않다면 없는 내용을 지어내는 것보다 모른다고 답하는 것이 더 낫습니다."
    ),
    ("human", "다음 질문에 대하여 한국어로 답변해주세요. {question}")
]

prompt = ChatPromptTemplate.from_messages(messages_with_questions)


def query_loop():
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOllama(model="gemma3", temperature=0.7)
    qa_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    while True:
        question = input("질문을 입력해주세요 (종료를 원하시면 'q'를 입력하세요.): ")
        if question == 'q':
            sys.exit(1)
            break
        else:
            result = qa_chain.invoke(question)
            print(result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python RAG_practice.py [build | check | reset | query]")
        sys.exit(1)
    cmd = sys.argv[1]

    if cmd == "build":
        build_vectorstore()
    elif cmd == "check":
        check_vectorstore_contents()
    elif cmd == "reset":
        reset_vectorstore()
    elif cmd == "query":
        query_loop()
    else:
        print("Unknown command:", cmd)
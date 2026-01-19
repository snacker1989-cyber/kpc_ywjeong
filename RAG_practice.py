import os
import sys
import subprocess

from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

DOCS_DIR = "pdfs"
VSTORE_DIR = "vectorstore"

def load_and_split(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)


def build_vectorstore(batch_size: int = 100):
    embeddings = OllamaEmbeddings(model="embeddinggemma")
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

def get_embeddings():
    emb = OllamaEmbeddings(model="embeddinggemma")
    return emb

def check_vectorstore_contents() -> None:
    """
    벡터 저장소에 저장된 PDF 파일 목록과 청크 수를 확인합니다.
    """
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    
    # 전체 문서 개수
    all_docs = db.get()
    
    if not all_docs or not all_docs['documents']:
        print("벡터 저장소가 비어있습니다.")
        return
    
    # 메타데이터에서 source 정보 추출
    sources = {}
    for metadata in all_docs['metadatas']:
        source = metadata.get('source', 'Unknown')
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    print("\n=== 벡터 저장소 현황 ===")
    print(f"총 임베딩된 청크 수: {len(all_docs['documents'])}")
    print("\n임베딩된 PDF 파일 목록:")
    for source, count in sources.items():
        pdf_name = Path(source).name
        print(f"  - {pdf_name}: {count}개 청크")
    print()

def reset_vectorstore() -> None:
    """
    벡터 저장소를 초기화합니다.
    """
    import shutil
    
    vstore_path = Path(VSTORE_DIR)
    
    if not vstore_path.exists():
        print(f"벡터 저장소가 존재하지 않습니다: {VSTORE_DIR}")
        return
    
    try:
        shutil.rmtree(vstore_path)
        print(f"✓ 벡터 저장소가 초기화되었습니다: {VSTORE_DIR}")
    except Exception as e:
        print(f"✗ 벡터 저장소 초기화 실패: {e}")

def query_loop(question: str):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOllama(model="gemma3", temperature=0.7)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = qa_chain.invoke(question)
    print("Answer:", answer["result"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python RAG_practice.py [build | check | reset | query \"your question\"]")
        sys.exit(1)
    cmd = sys.argv[1]

    if cmd == "build":
        build_vectorstore()
    elif cmd == "check":
        check_vectorstore_contents()
    elif cmd == "reset":
        reset_vectorstore()
    elif cmd == "query":
        if len(sys.argv) < 3:
            print("Provide a question: python RAG_practice.py query \"질문\"")
            sys.exit(1)
        query_loop(sys.argv[2])
    else:
        print("Unknown command:", cmd)
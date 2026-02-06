import os
import sys

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

DOCS_DIR = "data"
VSTORE_DIR = "vectorstore"

def load_and_split(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150,
                                               separators=["조(", "\n제", "\n\n", "\n", " ", ""])
    return splitter.split_documents(docs)
    ######### 어떻게 split되고 있는지 체크해보자

def get_embeddings():
    emb = OllamaEmbeddings(model="embeddinggemma")
    #emb = OllamaEmbeddings(model="qwen3-embedding")
    return emb

def build_vectorstore(batch_size: int = 30):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)

    total_indexed = 0
    pdfs = list(Path(DOCS_DIR).glob("*.pdf"))
    ####### *.pdf인지 *.xlsx/*xls인지 체크하고, 로더 분기 처리
    ####### 파일의 확장자에 따라 로더를 교체해서 쓰는 방식으로 (gemini한테 물어봐야하나...)

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
    ######## 메타데이터에서 source 정보 추출하는 부분 로그 확인, 필요 시 개선
    
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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python RAGbuilder.py [build | check | reset]")
        sys.exit(1)
    cmd = sys.argv[1]

    if cmd == "build":
        build_vectorstore()
    elif cmd == "check":
        check_vectorstore_contents()
    elif cmd == "reset":
        reset_vectorstore()
    else:
        print("Unknown command:", cmd)
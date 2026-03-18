import sys

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


DOCS_DIR = "data"
VSTORE_DIR = "vectorstore"
EMBEDDINGMODEL = "qwen3-embedding:4b"      # qwen3-embedding:4b vs embeddinggemma


def load_and_split(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, is_separator_regex=True,
                                               separators=[r"\n\n", r"(?=제\d+조\()", r"\n", r"\.", r"\s+", ""])
    return splitter.split_documents(docs)


def load_and_split_md(file_path):
    loader = TextLoader(file_path, encoding='utf-8')
    data = loader.load()
    content = data[0].page_content

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    # 1차 분할: 헤더 정보 기준으로 분할하고, 헤더 정보를 메타데이터로 저장
    md_header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,)
    md_header_splits = md_header_splitter.split_text(content)

    # 2차 분할: 글자 수 기준으로 세부 청킹
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150, is_separator_regex=True,
        separators=[r"\n\n", r"(?=제\d+조\()", r"\n", r"\.", r"\s+", ""])
    
    # 헤더로 나뉜 문서(1차 분할)를 글자 수 기준으로 최종 분할 (2차 분할)
    final_splits = text_splitter.split_documents(md_header_splits)
    return final_splits


def get_embeddings():
    emb = OllamaEmbeddings(model=EMBEDDINGMODEL)
    return emb

def build_vectorstore(batch_size: int = 30):
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)

    total_indexed = 0
    data = list(Path(DOCS_DIR).glob("*.md"))
    ####### *.pdf인지 *.xlsx/*xls인지 체크하고, 로더 분기 처리 - 파일의 확장자에 따라 로더를 교체해서 쓰는 방식으로 (gemini한테 물어봐야하나...)
    ####### 이건 지금 고민할 단계는 아닌거같다....

    if not data:
        print("No Markdown files found in:", DOCS_DIR)
        return
    
    for p in data:
        print("Processing:", p)
        docs = load_and_split_md(p)
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


def check_vectorstore_contents() -> None:       # 벡터 저장소에 저장된 Markdown 파일 목록과 청크 수 확인
    embeddings = get_embeddings()
    db = Chroma(persist_directory=VSTORE_DIR, embedding_function=embeddings)
    
    # 벡터 저장소에 저장된 문서의 수 확인
    all_docs = db.get()
    ###### get_retriever()에서 metadata를 따로 불러와서 체크 한 번 해보기
    
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
    print("\nEmbedded Markdown files:")
    for source, count in sources.items():
        data_name = Path(source).name
        print(f"  - {data_name}: {count} chunks")
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
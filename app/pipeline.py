# Script that clones a repo, understands structure and uploads smart chunks into pinecone
import os
import re
import stat
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.JS,
    ".md": Language.MARKDOWN,
}

def url_to_namespace(url):
    # Ensuring a correct namespace when given the url, 
    slug = re.sub(r"[^a-z0-9]+", "-", url.lower().removesuffix(".git").strip("/"))
    return slug.strip("-")[:62]

def is_indexed(index, namespace):
    stats = index.describe_index_stats()
    ns = stats.namespaces.get(namespace)
    return ns is not None and ns.vector_count > 0

def pinecone_setup():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=3072, # gemini-embedding-001 output dimensions
            metric="dotproduct", # hybrid search (dense + sparse)
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print("Index Created")

    return pc.Index(index_name)

def clone_repo(url, path):
    if os.path.exists(path):
        def _remove_readonly(func, path, _):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        shutil.rmtree(path, onexc=_remove_readonly)

    try:
        print("Started cloning")
        repo = Repo.clone_from(url, path)
        print("Finished cloning")
        return repo
    except Exception as e:
        print(f"Failed to clone: {e}")
        raise

def process_files(path):
    supported_extensions = tuple(EXTENSION_TO_LANGUAGE.keys())
    file_paths = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if filename.endswith(supported_extensions):
                file_paths.append(os.path.join(root, filename))
    return file_paths

def chunk_files(file_paths):
    chunks = []
    for filepath in file_paths:
        ext = os.path.splitext(filepath)[1]
        language = EXTENSION_TO_LANGUAGE.get(ext, Language.MARKDOWN)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=1000,
            chunk_overlap=100,
        )
        for i, chunk_text in enumerate(splitter.split_text(content)):
            chunks.append({
                "id": f"{filepath}_{i}",
                "file": filepath,
                "content": chunk_text,
            })

    print(f"Created {len(chunks)} chunks from {len(file_paths)} files")
    return chunks

# INCREASING THE EFFICIENCY - RATHER ONE API CALL 8 AT ONCE 
EMBED_BATCH_SIZE = 50   # chunks per Gemini API call
EMBED_WORKERS = 8       # concurrent embedding requests
UPSERT_WORKERS = 4      # concurrent Pinecone upsert requests

def _embed_batch(args):
    embedder, texts, start = args
    return start, embedder.embed_documents(texts)

def _build_records(chunks, dense_vectors, sparse_vectors):
    return [
        {
            "id": chunk["id"],
            "values": dense,
            "sparse_values": sparse,
            "metadata": {"file": chunk["file"], "content": chunk["content"]},
        }
        for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors)
    ]

def embed_and_upsert(index, chunks, namespace, bm25_path="bm25.json"):
    texts = [c["content"] for c in chunks]

    print("Fitting BM25 encoder...")
    encoder = BM25Encoder()
    encoder.fit(texts)
    encoder.dump(bm25_path)
    sparse_vectors = encoder.encode_documents(texts)

    print("Generating dense embeddings...")
    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    dense_vectors = [None] * len(texts)

    batches = [
        (embedder, texts[i:i + EMBED_BATCH_SIZE], i)
        for i in range(0, len(texts), EMBED_BATCH_SIZE)
    ]
    with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
        futures = {pool.submit(_embed_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(batches), desc="Embedding"):
            start, vecs = fut.result()
            dense_vectors[start:start + len(vecs)] = vecs

    print("Upserting to Pinecone...")
    upsert_batch_size = 100
    upsert_batches = [
        _build_records(
            chunks[i:i + upsert_batch_size],
            dense_vectors[i:i + upsert_batch_size],
            sparse_vectors[i:i + upsert_batch_size],
        )
        for i in range(0, len(chunks), upsert_batch_size)
    ]

    with ThreadPoolExecutor(max_workers=UPSERT_WORKERS) as pool:
        futures = {pool.submit(index.upsert, vectors=b, namespace=namespace): b for b in upsert_batches}
        for fut in tqdm(as_completed(futures), total=len(upsert_batches), desc="Upserting"):
            fut.result()

    print(f"Upserted {len(chunks)} vectors into namespace '{namespace}'")

def query_index(index, query, namespace, top_k=5, bm25_path="bm25.json"):
    embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    dense = embedder.embed_query(query)

    encoder = BM25Encoder()
    encoder.load(bm25_path)
    sparse = encoder.encode_queries(query)

    result = index.query(
        vector=dense,
        sparse_vector=sparse,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    return [
        {
            "file": match.metadata["file"],
            "content": match.metadata["content"],
            "score": match.score,
        }
        for match in result.matches
    ]

def generate_answer(query, results, history=None):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    context = "\n\n".join(f"File: {r['file']}\n{r['content']}" for r in results)
    messages = []
    if history:
        for msg in history[:-1]:  # exclude the current user message
            cls = HumanMessage if msg["role"] == "user" else AIMessage
            messages.append(cls(content=msg["content"]))
    messages.append(HumanMessage(content=(
        f"You are a code assistant. Use the following code snippets to answer the question.\n\n"
        f"{context}\n\nQuestion: {query}"
    )))
    return llm.invoke(messages).content

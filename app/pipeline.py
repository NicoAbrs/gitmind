# Script that clones a repo, understands structure and uploads smart chunks into pinecone
import os
import sys
import shutil
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from tqdm import tqdm
import time

load_dotenv()

EXTENSION_TO_LANGUAGE = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.JS,
    ".md": Language.MARKDOWN,
}

def pinecone_setup():
    # Sets up the pinecone index
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if there is a pinecone index, create one if there is not
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            vector_type="dense",
            dimensions=768, # text-embedding-004 output dimensions
            metric="dotproduct", # hybrid search (dense + sparse)
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print("Index Created")

    return pc.Index(index_name)

def clone_repo(url, path):
    # Wipes any existing path then clones the repo
    if os.path.exists(path):
        shutil.rmtree(path) # Clears up the previous run

    # Error catching as there are some read-only files in the git files.
    try:
        print("Started cloning")
        repo = Repo.clone_from(url, path)
        print("Finished cloning")
        return repo
    except Exception as e:
        print(f"Failed to clone: {e}")
        raise

def process_files(path):
    # Function deals with parsing of valid files within the cloned repo
    supported_extensions = tuple(EXTENSION_TO_LANGUAGE.keys())
    file_paths = []

    # Looping through the path, append valid file path with supported extensions
    for root, dirs, files in os.walk(path):
        # Check if the files are a supported extension
        for filename in files:
            if filename.endswith(supported_extensions):
                # Appending the files
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

def embed_and_upsert(index, chunks, bm25_path="bm25.json"):
    texts = [c["content"] for c in chunks]

    print("Fitting BM25 encoder...")
    encoder = BM25Encoder()
    encoder.fit(texts)
    encoder.dump(bm25_path)
    sparse_vectors = encoder.encode_documents(texts)

    print("Generating dense embeddings...")
    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    dense_vectors = embedder.embed_documents(texts)

    batch_size = 100
    print("Upserting to Pinecone...")
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch_chunks = chunks[i:i + batch_size]
        batch_dense = dense_vectors[i:i + batch_size]
        batch_sparse = sparse_vectors[i:i + batch_size]

        records = [
            {
                "id": chunk["id"],
                "values": dense,
                "sparse_values": sparse,
                "metadata": {"file": chunk["file"], "content": chunk["content"]},
            }
            for chunk, dense, sparse in zip(batch_chunks, batch_dense, batch_sparse)
        ]
        index.upsert(vectors=records)

    print(f"Upserted {len(chunks)} vectors")

def query_index(index, query, top_k=5, bm25_path="bm25.json"):
    embedder = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    dense = embedder.embed_query(query)

    encoder = BM25Encoder.load(bm25_path)
    sparse = encoder.encode_queries(query)

    result = index.query(
        vector=dense,
        sparse_vector=sparse,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {
            "file": match.metadata["file"],
            "content": match.metadata["content"],
            "score": match.score,
        }
        for match in result.matches
    ]

def generate_answer(query, results):
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    context = "\n\n".join(f"File: {r['file']}\n{r['content']}" for r in results)
    prompt = (
        f"You are a code assistant. Use the following code snippets to answer the question.\n\n"
        f"{context}\n\n"
        f"Question: {query}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

def main(repo_url, query, clone_path="./cloned_repo"):
    index = pinecone_setup()
    clone_repo(repo_url, clone_path)
    file_paths = process_files(clone_path)
    chunks = chunk_files(file_paths)
    embed_and_upsert(index, chunks)
    results = query_index(index, query)
    answer = generate_answer(query, results)

    print(f"\n--- Answer for: '{query}' ---\n")
    print(answer)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python app/pipeline.py <repo_url> \"<query>\"")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

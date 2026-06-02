import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from app.pipeline import (
    pinecone_setup,
    clone_repo,
    process_files,
    chunk_files,
    embed_and_upsert,
    query_index,
    generate_answer,
    url_to_namespace,
    is_indexed,
)

st.set_page_config(page_title="GitMind", page_icon="🧠", layout="wide")
st.title("🧠 GitMind — Ask Your Codebase")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "namespace" not in st.session_state:
    st.session_state.namespace = None
if "active_repo" not in st.session_state:
    st.session_state.active_repo = None

with st.sidebar:
    st.header("Index a Repository")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")

    if st.button("Load Repository", type="primary", disabled=not repo_url):
        namespace = url_to_namespace(repo_url)

        # Reset chat whenever repo changes
        if repo_url != st.session_state.active_repo:
            st.session_state.messages = []

        with st.status("Loading repository...", expanded=True) as status:
            try:
                st.write("Connecting to Pinecone...")
                index = pinecone_setup()

                if is_indexed(index, namespace):
                    st.write("Repository already indexed — loading from cache...")
                    # BM25 encoder must be re-fit; re-clone and chunk to rebuild it
                    # without re-uploading to Pinecone
                    st.write("Rebuilding sparse encoder from repo...")
                    clone_repo(repo_url, "./cloned_repo")
                    file_paths = process_files("./cloned_repo")
                    chunks = chunk_files(file_paths)
                    from pinecone_text.sparse import BM25Encoder
                    encoder = BM25Encoder()
                    encoder.fit([c["content"] for c in chunks])
                    encoder.dump("bm25.json")
                else:
                    st.write("Cloning repository...")
                    clone_repo(repo_url, "./cloned_repo")

                    st.write("Discovering files...")
                    file_paths = process_files("./cloned_repo")
                    st.write(f"Found {len(file_paths)} files")

                    st.write("Chunking files...")
                    chunks = chunk_files(file_paths)

                    st.write(f"Embedding and uploading {len(chunks)} chunks...")
                    embed_and_upsert(index, chunks, namespace=namespace)

                st.session_state.index = index
                st.session_state.namespace = namespace
                st.session_state.active_repo = repo_url
                status.update(label="Ready to chat!", state="complete")

            except Exception as e:
                status.update(label="Failed", state="error")
                st.error(str(e))

    if st.session_state.active_repo:
        st.success(f"Active: `{st.session_state.active_repo}`")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if st.session_state.namespace:
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask something about the codebase..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                results = query_index(st.session_state.index, prompt, namespace=st.session_state.namespace)
                answer = generate_answer(prompt, results, history=st.session_state.messages)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Enter a repository URL in the sidebar and click **Load Repository** to begin.")

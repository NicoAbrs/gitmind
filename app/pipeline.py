# Script that clones a repo, understands structure and uploads smart chunks into pinecone
import os
import shutil
from git import Repo
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import time

def pinecone_setup(): 
    # Sets up the pinecone index
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Check if there is a pinecone index, create one if there is not
    if not pc.has_index(index_name): 
        pc.create_index(
            name=index_name, 
            vector_type="dense",
            dimensions=1536, # Match the input dimensions for the openai model, change if needed 
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
        print("Failed to clone: {e}")
        raise 

def process_files(path): 
    # Function deals with parsing of valid files within the cloned repo
    supported_extensions = (".py", ".js", ".ts", ".md")
    file_paths = {}
    
    # Looping through the path, append valid file path with supported extensions
    for root, dirs, files in os.walk(path): 
        # Check if the files are a supported extension 
        for filename in files: 
            if filename.endswith(supported_extensions): 
                # Appending the files
                file_paths.append(os.path.join(root, filename))

    return file_paths

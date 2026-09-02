# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    WebBaseLoader,
    UnstructuredPDFLoader,
    TextLoader,
    CSVLoader 
)
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import hashlib
import os
import shutil
import mimetypes
import time



# Handling which API to use based on public vs NVIDIA internal
# Check if the INTERNAL_API environment variable is set to yes. Don't set if not NVIDIA employee, as you can't access them.
INTERNAL_API = os.getenv('INTERNAL_API', 'no')

# Default model for public embedding
EMBEDDINGS_MODEL = 'nvidia/nemotron-3-embed-1b'

# Set the chunk size and overlap for the text splitter. Uses defaults but allows them to be set as environment variables.
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 0

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))


if INTERNAL_API == 'yes':
    # NVIDIA employees can use internal endpoints
    EMBEDDINGS_MODEL = 'nvdev/nvidia/llama-nemotron-embed-1b-v2'
    print("[config] INTERNAL_API detected.")
    print(f"[config] Using internal embedding model: {EMBEDDINGS_MODEL}")

else:
    print("[config] No INTERNAL_API set.")
    print(f"[config] Using public embedding model: {EMBEDDINGS_MODEL}")


def _collection_name(embedding_model: str) -> str:
    if embedding_model == EMBEDDINGS_MODEL:
        return "rag-chroma"
    digest = hashlib.sha256(embedding_model.encode("utf-8")).hexdigest()[:12]
    return f"rag-chroma-{digest}"


# Adding nltk data
import nltk

def download_nltk_if_missing():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_if_missing()

    
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

# Functions for dealing with URLs
def is_valid_url(url: str) -> bool:
    """ This is a helper function for checking if the URL is valid. It isn't fail proof, but it will catch most common errors. """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def safe_load(url):
    """ This is a helper function for loading the URL. It protects against false negatives from is_value_url and 
        filters for actual web pages. Returns None if it fails.
    """
    try:
        return WebBaseLoader(url).load()
    except Exception as e:
        print(f"[Documents] ⚠ Could not load {url}: {e}")
        return None


def upload(urls: List[str], embedding_model: str = EMBEDDINGS_MODEL):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """

    urls = [url.strip() for url in urls if url.strip()]
    for url in urls:
        if not is_valid_url(url):
            print(f"[Documents] ⚠ Skipping invalid URL: {url}")
    urls = [url for url in urls if is_valid_url(url)]

    if not urls:
        print("[Documents] ✗ No valid URLs provided — nothing was added to the context.")
        return None

    print(f"[Documents] Adding {len(urls)} webpage(s) to the context...")
    docs = []
    for url in urls:
        result = safe_load(url)
        if result is not None:
            docs.append(result)
            print(f"[Documents] ✓ Loaded {url}")

    docs_list = [item for sublist in docs for item in sublist]

    if not docs_list:
        # If no documents were loaded, return None
        print("[Documents] ✗ None of the webpages could be loaded — nothing was added to the context.")
        return None

    print(f"[Documents] Loaded {len(docs)} of {len(urls)} webpage(s) successfully")

    try:
        doc_splits = split_documents(docs_list)
        return embed_documents(doc_splits, embedding_model)

    except Exception as e:
        print(f"[Documents] ✗ Failed to add webpages to the context: {e}")
        return None


# Functions for dealing with file uploads/embeddings

## Helper functions

def load_documents_from_files(file_paths: List[str]) -> List[Any]:
    """Load and return documents from supported file types."""
    docs = []

    for fpath in file_paths:
        ext = os.path.splitext(fpath)[-1].lower()

        loader_cls = {
            ".pdf": UnstructuredPDFLoader,
            ".txt": TextLoader,
            ".md": TextLoader,
            ".csv": CSVLoader
        }.get(ext)

        if loader_cls is None:
            print(f"[Documents] ⚠ Skipping unsupported file type: {os.path.basename(fpath)} (supported: .pdf, .txt, .md, .csv)")
            continue

        try:
            loaded = loader_cls(fpath).load()
            docs.append(loaded)
            print(f"[Documents] ✓ Loaded {os.path.basename(fpath)}")
        except Exception as e:
            print(f"[Documents] ✗ Failed to load {os.path.basename(fpath)}: {e}")

    return [item for sublist in docs for item in sublist]


def split_documents(docs: List[Any]):
    """Split documents into smaller chunks using recursive splitter."""
    print(f"[Documents] Splitting {len(docs)} document(s) into chunks of ~{CHUNK_SIZE} tokens ({CHUNK_OVERLAP} token overlap)...")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    doc_splits = splitter.split_documents(docs)
    print(f"[Documents] Created {len(doc_splits)} chunks")
    return doc_splits

def embed_documents(
    doc_splits: List[Any],
    embedding_model: str = EMBEDDINGS_MODEL,
):
    """Embed and store the split documents into Chroma vectorstore."""
    try:
        print(f"[Documents] Embedding {len(doc_splits)} chunks with {embedding_model}...")
        start = time.time()

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=_collection_name(embedding_model),
            embedding=NVIDIAEmbeddings(model=embedding_model),
            persist_directory="/project/data",
        )
        print(f"[Documents] ✓ Context ready — {len(doc_splits)} chunks stored in the vector database ({time.time() - start:.1f}s)")
        return vectorstore

    except Exception as e:
        print(f"[Documents] ✗ Embedding failed — could not build the context: {e}")
        return None


## Main function that use helper functions 

def upload_files(
    file_paths: List[str],
    embedding_model: str = EMBEDDINGS_MODEL,
):
    """Upload files into the vector store pipeline."""

    if not file_paths:
        print("[Documents] ✗ No files provided — nothing was added to the context.")
        return None

    print(f"[Documents] Adding {len(file_paths)} file(s) to the context...")

    try:
        docs_list = load_documents_from_files(file_paths)

        if not docs_list:
            print("[Documents] ✗ None of the files could be loaded — nothing was added to the context.")
            return None

        doc_splits = split_documents(docs_list)
        return embed_documents(doc_splits, embedding_model)

    except Exception as e:
        print(f"[Documents] ✗ Failed to add files to the context: {e}")
        return None




def _clear(
    embedding_model: str = EMBEDDINGS_MODEL,
    persist_directory: str = "/project/data",
    collection_name: Optional[str] = None,
    delete_all: bool = False,
):
    """Clear the Chroma collection and optionally delete all shard folders (excluding hidden files)."""
    try:
        print("[Documents] Clearing the context...")
        collection_name = collection_name or _collection_name(embedding_model)

        # Clear the collection via Chroma client
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=NVIDIAEmbeddings(model=embedding_model),
            persist_directory=persist_directory,
        )
        vectorstore._client.delete_collection(name=collection_name)
        vectorstore._client.create_collection(name=collection_name)

        if delete_all:
            removed = 0
            for item in os.listdir(persist_directory):
                if item.startswith(".") or item.startswith("chroma.") or item.startswith("readme-images"):
                    continue  # Skip hidden files like .gitkeep and the sqlite3 file

                path = os.path.join(persist_directory, item)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                        removed += 1
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                        removed += 1
                except Exception as file_err:
                    print(f"[Documents] ⚠ Could not delete cached index data '{item}': {file_err}")
            if removed:
                print(f"[Documents] Removed {removed} cached index item(s) from disk")

        print("[Documents] ✓ Context cleared — the vector database is now empty")

    except Exception as e:
        print(f"[Documents] ✗ Failed to clear the context: {e}")


      
# def clear():
#     """ This is a helper function for emptying the collection the vector store. """
#     vectorstore = Chroma(
#         collection_name="rag-chroma",
#         embedding_function=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
#         persist_directory="/project/data",
#     )
    
#     vectorstore._client.delete_collection(name="rag-chroma")
#     vectorstore._client.create_collection(name="rag-chroma")

def get_retriever(embedding_model: str = EMBEDDINGS_MODEL):
    """ This is a helper function for returning the retriever object of the vector store. """
    vectorstore = Chroma(
        collection_name=_collection_name(embedding_model),
        embedding_function=NVIDIAEmbeddings(model=embedding_model),
        persist_directory="/project/data",
    )
    retriever = vectorstore.as_retriever()
    return retriever

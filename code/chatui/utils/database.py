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

from typing import Any, Dict, List, Tuple, Union
from urllib.parse import urlparse
import os
import mimetypes

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
# from langchain_community.vectorstores import Chroma
# from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
# from typing import Any, Dict, List, Tuple, Union
# from urllib.parse import urlparse

# import os

# Handling which API to use based on public vs NVIDIA internal
# Check if the INTERNAL_API environment variable is set. Don't set if not NVIDIA employee, as you can't access them.
INTERNAL_API = os.getenv('INTERNAL_API', '')

# Default model for public embedding
EMBEDDINGS_MODEL = 'NV-Embed-QA'

if INTERNAL_API != '':
    # NVIDIA employees must use internal endpoints
    EMBEDDINGS_MODEL = 'nvdev/nvidia/nv-embedqa-e5-v5'
    print("[config] INTERNAL_API detected.")
    print(f"[config] Using internal embedding model: {EMBEDDINGS_MODEL}")

else:
    print("[config] No INTERNAL_API set.")
    print(f"[config] Using public embedding model: {EMBEDDINGS_MODEL}")

# Set the chunk size and overlap for the text splitter. Uses defaults but allows them to be set as environment variables.
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 0

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", DEFAULT_CHUNK_SIZE))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP))


# Download nltk data
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

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
        print(f"[upload] Skipping {url}: {e}")
        return None


def upload(urls: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """

    urls = [url for url in urls if is_valid_url(url)]

    docs = []
    for url in urls:
        result = safe_load(url)
        if result is not None:
            docs.append(result)

    docs_list = [item for sublist in docs for item in sublist]

    if not docs_list:
        # If no documents were loaded, return None
        print("[upload] No URLs provided.")
        return None
    
    try:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs_list)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
            persist_directory="/project/data",
        )
        return vectorstore

    except Exception as e:
        print(f"[upload] Vectorstore creation failed: {e}")
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
            print(f"[load_documents] Skipping unsupported file type: {fpath}")
            continue

        try:
            loaded = loader_cls(fpath).load()
            docs.append(loaded)
        except Exception as e:
            print(f"[load_documents] Failed to load {fpath}: {e}")

    return [item for sublist in docs for item in sublist]


def split_documents(docs: List[Any]):
    """Split documents into smaller chunks using recursive splitter."""
    print(f"[split_documents] Splitting {len(docs)} docs with chunk size {CHUNK_SIZE}, overlap {CHUNK_OVERLAP}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)

def embed_documents(doc_splits: List[Any]):
    """Embed and store the split documents into Chroma vectorstore."""
    try:
        print(f"[embed_documents] Embedding {len(doc_splits)} chunks using model: {EMBEDDINGS_MODEL}")

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
            persist_directory="/project/data",
        )
        return vectorstore

    except Exception as e:
        print(f"[embed_documents] Vectorstore creation failed: {e}")
        return None


## Main function that use helper functions 

def upload_files(file_paths: List[str]):
    """Upload files into the vector store pipeline."""

    if not file_paths:
        print("[upload_files] No documents provided.")
        return None

    try:
        docs_list = load_documents_from_files(file_paths)

        if not docs_list:
            print("[upload_files] No documents successfully loaded.")
            return None

        doc_splits = split_documents(docs_list)
        return embed_documents(doc_splits)

    except Exception as e:
        print(f"[upload_files] Pipeline failed: {e}")
        return None

        
# def upload_pdf(documents: List[str]):
#     """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """

#     if not docs:
#         print("[upload] No URLs provided.")
#         return None

#     docs = [UnstructuredPDFLoader(document).load() for document in documents]
#     docs_list = [item for sublist in docs for item in sublist]
    
#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=CHUNK_SIZE, chunk_overlap=0
#     )
#     doc_splits = text_splitter.split_documents(docs_list)
    
#     # Add to vectorDB
#     vectorstore = Chroma.from_documents(
#         documents=doc_splits,
#         collection_name="rag-chroma",
#         embedding=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
#         persist_directory="/project/data",
#     )
#     return vectorstore

def clear():
    """ This is a helper function for emptying the collection the vector store. """
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
        persist_directory="/project/data",
    )
    
    vectorstore._client.delete_collection(name="rag-chroma")
    vectorstore._client.create_collection(name="rag-chroma")

def get_retriever(): 
    """ This is a helper function for returning the retriever object of the vector store. """
    vectorstore = Chroma(
        collection_name="rag-chroma",
        embedding_function=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
        persist_directory="/project/data",
    )
    retriever = vectorstore.as_retriever()
    return retriever

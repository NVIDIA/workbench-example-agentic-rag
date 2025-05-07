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
from langchain_community.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from typing import Any, Dict, List, Tuple, Union

import os
INTERNAL_API = os.getenv('INTERNAL_API', '')

# Set the embeddings model target
EMBEDDINGS_MODEL = 'NV-Embed-QA'

#
if INTERNAL_API != '':
    EMBEDDINGS_MODEL = 'nvdev/nvidia/nv-embedqa-e5-v5'

# Download nltk data
import nltk
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")


def upload(urls: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
        persist_directory="/project/data",
    )
    return vectorstore

def upload_pdf(documents: List[str]):
    """ This is a helper function for parsing the user inputted URLs and uploading them into the vector store. """
    docs = [UnstructuredPDFLoader(document).load() for document in documents]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # Add to vectorDB
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=NVIDIAEmbeddings(model=EMBEDDINGS_MODEL),
        persist_directory="/project/data",
    )
    return vectorstore

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

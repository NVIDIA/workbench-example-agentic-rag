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

""" These are the default prompts used by the applications for Mixtral/Mistral-based instruction models. """

router_prompt = """
<s>[INST] You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. Your response format is non-negotiable: you must return a JSON with a single key 'datasource' and no premable or explanation. 

Question to route: {question} [/INST]
"""

retrieval_prompt = """
<s>[INST] You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
Your response format is non-negotiable: you must provide the binary score as a JSON with a single key 'score' and no premable or explanation.

Here is the retrieved document: \n {document} \n
Here is the user question: {question} 

[/INST]
"""

generator_prompt = """
<s>[INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise 

Question: {question} 
Context: {context} 

Answer: [/INST]
"""

hallucination_prompt = """
<s>[INST] You are a grader assessing whether an answer is grounded in and supported by a set of facts. Give a binary 'yes' or 'no' score to indicate whether the answer is grounded in and supported by a set of facts. Your response format is non-negotiable: you must provide the binary score as a JSON with a single key 'score' and no preamble or explanation. 

Here are the facts:
\n ------- \n {documents} \n ------- \n
Here is the answer: {generation} 

[/INST]
"""

answer_prompt = """
<s>[INST] You are a grader assessing whether an answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. Your response format is non-negotiable: you must provide the binary score as a JSON with a single key 'score' and no preamble or explanation.

Here is the answer:
\n ------- \n {generation} \n ------- \n
Here is the question: {question} 

[/INST]
"""

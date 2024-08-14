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

from langgraph.graph import END, StateGraph
from chatui.utils import graph

def compile_graph():
    
    # Compile the agent graph
    workflow = StateGraph(graph.GraphState)
    
    # Define the nodes
    workflow.add_node("websearch", graph.web_search)  # web search
    workflow.add_node("retrieve", graph.retrieve)  # retrieve
    workflow.add_node("grade_documents", graph.grade_documents)  # grade documents
    workflow.add_node("generate", graph.generate)  # generate

    # Build graph
    workflow.set_conditional_entry_point(
        graph.route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
    )
    
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        graph.decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        graph.grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    return workflow

    
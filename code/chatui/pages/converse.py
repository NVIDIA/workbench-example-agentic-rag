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

### This module contains the chatui gui for having a conversation. ###

import functools
from typing import Any, Dict, List, Tuple, Union

import gradio as gr
import shutil
import os
import subprocess
import sys
import json

from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError 

from requests.exceptions import HTTPError
import traceback

from langchain_core.exceptions import OutputParserException

from chatui.utils.error_messages import QUERY_ERROR_MESSAGES
from chatui.utils.graph import TavilyAPIError

# UI names and labels
SELF_HOSTED_TAB_NAME = "Self-Hosted Endpoint"
HOST_NAME = "Local NIM or Remote IP/Hostname"
HOST_PORT = "Host Port"
HOST_MODEL = "Model Name"


# Set recursion limit 
DEFAULT_RECURSION_LIMIT = 10
RECURSION_LIMIT = int(os.getenv("RECURSION_LIMIT", DEFAULT_RECURSION_LIMIT))


# Model identifiers with prefix
NANO = "nvidia/nemotron-3-nano-30b-a3b"
SUPER = "nvidia/nemotron-3-super-120b-a12b"
ULTRA = "nvidia/nemotron-3-ultra-550b-a55b"

# check if the internal API is set
INTERNAL_API = os.getenv('INTERNAL_API', 'no')

# Modify model identifiers (to use the internal endpoints if that variable is set).
if INTERNAL_API == 'yes':
    NANO = 'nvdev/nvidia/nemotron-3-nano-30b-a3b'
    SUPER = 'nvdev/nvidia/nemotron-3-super-120b-a12b'
    ULTRA = 'nvdev/nvidia/nemotron-3-ultra-550b-a55b'

# Model presets applied to the five API-endpoint dropdowns at once
# (order: router, retrieval grader, generator, hallucination grader, answer grader).
PRESET_FAST = "⚡ Fast — Nano for every component"
PRESET_BALANCED = "⚖️ Balanced — Super for every component"
PRESET_QUALITY = "🏆 Max quality — Ultra generator, Super for the rest"
MODEL_PRESETS = {
    PRESET_FAST: (NANO, NANO, NANO, NANO, NANO),
    PRESET_BALANCED: (SUPER, SUPER, SUPER, SUPER, SUPER),
    PRESET_QUALITY: (SUPER, SUPER, ULTRA, SUPER, SUPER),
}

# URLs for default example docs for the RAG.
doc_links = (
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/introduction.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/desktop-app.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/command-line-interface.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/installation-overview.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/desktop-app-install.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/full-local-install.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/remote-install.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/uninstall.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/install/update.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/overview/onboarding-project.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/project-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/location-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/single-container-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/compose-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/application-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/versioning-concept.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/concepts/understand-project-specification.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/create-clone-publish.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/file-browser.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/deep-linking.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/projects/versioning.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/add-existing-location.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/add-brev.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/ides/vs-code.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/ides/cursor.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/ides/windsurf.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/package-management.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/prebuild-script.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/postbuild-script.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/runtime-configuration.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/hardware.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/environments/multi-container.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/github-gitlab.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/self-hosted-gitlab.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/integrations/nvidia-integrations.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/app-sharing.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/use-custom-container.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/how-to/convert-repo.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/applications-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/user-interface/desktop-app.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/user-interface/cli.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/glossary.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/runtime-configuration-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/hardware-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/custom-container.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/compose-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/compose-patterns-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/spec.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/projects/base-environments.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/remote-locations.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/components.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/settings.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/customize-the-ui.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/runtimes.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/customca.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/workbench-application/proxy.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/support-matrix.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/versioning/git-configuration-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/windows-full-local-reference.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/troubleshooting/troubleshooting.html",
    "https://docs.nvidia.com/ai-workbench/user-guide/latest/reference/troubleshooting/logging.html",
)
EXAMPLE_LINKS_LEN = 10

EXAMPLE_LINKS = "\n".join(doc_links)

from chatui import assets, chat_client
from chatui.prompts import prompts_nemotron
from chatui.utils import compile, database, logger, gpu_compatibility

from langgraph.graph import END, StateGraph

PATH = "/"
TITLE = "Agentic RAG: Chat UI"
OUTPUT_TOKENS = 250
MAX_DOCS = 5


""" Environment + context status helpers, shown above the chat and in the Documents tab. """

def _status_strip_md() -> str:
    """One-line setup status: API keys and current context size."""
    if os.getenv("NVIDIA_API_KEY"):
        nvidia = "🔑 NVIDIA API key: set ✓"
    else:
        nvidia = "🔑 NVIDIA API key: **missing ✗** (set it in AI Workbench → Project Container → Variables)"
    if os.getenv("TAVILY_API_KEY"):
        tavily = "🌐 Tavily key: set ✓"
    else:
        tavily = "🌐 Tavily key: **missing ✗** (web search fallback will fail)"
    chunks, sources = database.get_context_summary()
    if chunks:
        context = f"📚 Context: {chunks} chunks from {len(sources)} source(s)"
    else:
        context = "📚 Context: empty — add documents in the Documents tab"
    return f"{nvidia} · {tavily} · {context}"


def _shorten(text: str, limit: int = 70) -> str:
    text = str(text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _context_panel_md() -> str:
    """Markdown summary of every source currently in the vector database."""
    chunks, sources = database.get_context_summary()
    if not chunks:
        return ("**The context is currently empty.** Add webpages or files below — "
                "until you do, document questions will fall back to web search.")
    lines = [f"**{chunks} chunks from {len(sources)} source(s) are in the context.**", ""]
    ranked = sorted(sources.items(), key=lambda item: -item[1])
    for source, count in ranked[:15]:
        if str(source).startswith("http"):
            label = f"[{_shorten(source)}]({source})"
        else:
            label = _shorten(os.path.basename(str(source)))
        lines.append(f"- {label} — {count} chunk(s)")
    if len(ranked) > 15:
        lines.append(f"- …and {len(ranked) - 15} more source(s)")
    return "\n".join(lines)


""" Helpers that translate LangGraph stream events into a user-facing agent timeline.

The graph emits one event per node. Grader verdicts are not separate events, but they
are fully inferable from node transitions: generate → generate means the groundedness
check failed (regenerate), generate → websearch means the answer check failed (retry
with web search), and reaching the end after generate means both checks passed.
"""

def _new_timeline() -> Dict[str, Any]:
    return {"steps": [], "retries": 0, "last_node": None, "retrieved": 0}


def _record_timeline_event(timeline: Dict[str, Any], node: str, delta: Dict[str, Any]) -> None:
    steps = timeline["steps"]
    last = timeline["last_node"]
    if node == "retrieve":
        steps.append("🗂️ **Router** — the question matches the document context, searching the vector database")
        timeline["retrieved"] = len(delta.get("documents") or [])
        steps.append(f"📚 **Retriever** — pulled {timeline['retrieved']} chunk(s) from the vector database")
    elif node == "grade_documents":
        kept = len(delta.get("documents") or [])
        total = timeline["retrieved"] or kept
        if delta.get("web_search") == "Yes":
            steps.append(f"🔍 **Relevance check** — none of the {total} chunk(s) were relevant, falling back to web search")
        else:
            steps.append(f"🔍 **Relevance check** — kept {kept} of {total} chunk(s)")
    elif node == "websearch":
        if last is None:
            steps.append("🗂️ **Router** — the question falls outside the document context, using web search")
        elif last == "generate":
            timeline["retries"] += 1
            steps.append("🛠️ **Answer check** — the draft did not address the question, self-correcting with a web search")
        steps.append("🌐 **Web search** — gathering live results")
    elif node == "generate":
        if last == "generate":
            timeline["retries"] += 1
            steps.append("🛠️ **Groundedness check** — the draft was not supported by the context, regenerating")
        steps.append("✍️ **Generator** — drafting an answer from the context")
    timeline["last_node"] = node


def _finish_timeline(timeline: Dict[str, Any]) -> None:
    """The graph only reaches END after both graders pass."""
    timeline["steps"].append("✅ **Groundedness check** — the answer is supported by the context")
    timeline["steps"].append("✅ **Answer check** — the answer addresses the question")


def _timeline_md(timeline: Dict[str, Any], working: bool = True) -> str:
    lines = [f"{i}. {step}" for i, step in enumerate(timeline["steps"], start=1)]
    if working:
        header = "🤖 **The agent is working through your question:**"
        lines.append(f"{len(lines) + 1}. ⏳ *working…*")
    else:
        header = "🤖 **What the agent tried:**"
    return header + "\n\n" + "\n".join(lines)


def _collect_sources(documents) -> List[Tuple[str, Union[str, None]]]:
    """Deduped (label, url) pairs for the documents behind an answer. Web-search results
    are synthesized without metadata, so any document with no source is the Tavily doc."""
    pairs = []
    seen = set()
    web_results = False
    for doc in documents or []:
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source")
        if not source:
            web_results = True
            continue
        if source in seen:
            continue
        seen.add(source)
        if str(source).startswith("http"):
            title = str(metadata.get("title") or source).strip() or str(source)
            title = _shorten(title.replace("[", "(").replace("]", ")"), 80)
            pairs.append((title, str(source)))
        else:
            pairs.append((os.path.basename(str(source)), None))
    if web_results:
        pairs.append(("Live web search results (Tavily)", None))
    return pairs


def _final_answer_md(generation: str, timeline: Dict[str, Any], documents) -> str:
    """Compose the final chat message: answer, verification badges, and sources."""
    badges = ["✅ Grounded in the context", "✅ Addresses the question"]
    if timeline["retries"]:
        plural = "s" if timeline["retries"] > 1 else ""
        badges.append(f"🔄 {timeline['retries']} self-correction{plural}")
    parts = [generation.strip(), "---", " · ".join(badges)]
    sources = _collect_sources(documents)
    if sources:
        shown = sources[:6]
        lines = [f"- [{label}]({url})" if url else f"- {label}" for label, url in shown]
        if len(sources) > len(shown):
            lines.append(f"- …and {len(sources) - len(shown)} more")
        parts.append("**Sources**\n" + "\n".join(lines))
    return "\n\n".join(parts)


def _build_trace(question: str, timeline: Dict[str, Any], documents=None, status: str = "in progress") -> Dict[str, Any]:
    """Curated run summary for the Monitor → Response Trace tab (no prompts or endpoint internals)."""
    return {
        "question": question,
        "status": status,
        "agent_steps": [step.replace("**", "") for step in timeline["steps"]],
        "self_corrections": timeline["retries"],
        "sources": [url or label for label, url in _collect_sources(documents)],
    }


def _refresh_context_displays():
    """Recompute the status strip and the Documents-tab context panel."""
    return _status_strip_md(), _context_panel_md()

### Load in CSS here for components that need custom styling. ###

_LOCAL_CSS = """
#contextbox {
    overflow-y: scroll !important;
    max-height: 400px;
}

#params .tabs {
    display: flex;
    flex-direction: column;
    flex-grow: 1;
}
#params .tabitem[style="display: block;"] {
    flex-grow: 1;
    display: flex !important;
}
#params .gap {
    flex-grow: 1;
}
#params .form {
    flex-grow: 1 !important;
}
#params .form > :last-child{
    flex-grow: 1;
}
#accordion {
}
#rag-inputs .svelte-1gfkn6j .svelte-s1r2yt .svelte-cmf5ev {
    /* Darker shade of NVIDIA green: #76b900 fails WCAG AA contrast for text on white */
    color: #4e7a00 !important;
}
.mode-banner {
    font-size: 1.05rem;
    font-weight: 500;
    background-color: #f0f4f8;
    padding: 0.5em 0.75em;
    border-left: 2px solid #76b900;
    margin-bottom: 0.5em;
    border-radius: 2px;
}
.status-strip {
    font-size: 0.9rem;
    background-color: #f0f4f8;
    padding: 0.4em 0.75em;
    border-left: 2px solid #76b900;
    border-radius: 2px;
}
.status-strip p {
    font-size: 0.9rem !important;
}
.sample-caption p {
    font-size: 0.78rem !important;
    color: #4a4a4a !important;
    margin-top: -4px;
}
.diagram-caption p {
    font-size: 0.85rem !important;
    color: #4a4a4a !important;
}
"""

sys.stdout = logger.Logger("/project/code/output.log")

def build_page(client: chat_client.ChatClient) -> gr.Blocks:
    """
    Build the gradio page to be mounted in the frame.
    
    Parameters: 
        client (chat_client.ChatClient): The chat client running the application. 
    
    Returns:
        page (gr.Blocks): A Gradio page.
    """
    kui_theme, kui_styles = assets.load_theme("kaizen")
    
    """ Compile the agentic graph. """
    
    workflow = compile.compile_graph()
    app = workflow.compile()

    """ List of currently supported models. """
    
    model_list = [ULTRA, SUPER, NANO]

    with gr.Blocks(title=TITLE, theme=kui_theme, css=kui_styles + _LOCAL_CSS) as page:
        gr.Markdown(f"# {TITLE}")

        """ Keep state of which queries need to use NIMs vs API Endpoints. """
        
        router_use_nim = gr.State(False)
        retrieval_use_nim = gr.State(False)
        generator_use_nim = gr.State(False)
        hallucination_use_nim = gr.State(False)
        answer_use_nim = gr.State(False)

        """ Build the Chat Application. """
        
        with gr.Row(equal_height=True):

            # Left Column will display the chatbot
            with gr.Column(scale=16, min_width=350):

                # Setup + context status, refreshed on page load and after document changes.
                with gr.Row():
                    status_strip = gr.Markdown("⏳ Checking your setup…", elem_classes=["status-strip"])

                # Main chatbot panel.
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=350):
                        chatbot = gr.Chatbot(show_label=False, height=500, show_copy_button=True)

                # Message box for user input
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3, min_width=450):
                        msg = gr.Textbox(
                            show_label=False,
                            placeholder="Enter text and press ENTER",
                            container=False,
                            interactive=True,
                        )

                    with gr.Column(scale=1, min_width=150):
                        _ = gr.ClearButton([msg, chatbot], value="Clear Chat History")

                # Sample questions, each labeled with the agent path it demonstrates.
                gr.Markdown("**Try a sample** — each one exercises a different agent path:")
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=220):
                        sample_query_1 = gr.Button("How do I add the GitHub integration using OAuth?", variant="secondary", size="sm", interactive=True)
                        gr.Markdown("📚 Answered from the document context", elem_classes=["sample-caption"])
                    with gr.Column(min_width=220):
                        sample_query_2 = gr.Button("What are the NVIDIA-provided default base environments?", variant="secondary", size="sm", interactive=True)
                        gr.Markdown("📚 Answered from the document context", elem_classes=["sample-caption"])
                with gr.Row(equal_height=True):
                    with gr.Column(min_width=220):
                        sample_query_3 = gr.Button("What are the top technology headlines today?", variant="secondary", size="sm", interactive=True)
                        gr.Markdown("🌐 Routed straight to web search", elem_classes=["sample-caption"])
                    with gr.Column(min_width=220):
                        sample_query_4 = gr.Button("How do I fix an inaccessible remote Location?", variant="secondary", size="sm", interactive=True)
                        gr.Markdown("📚 Tries the documents first — watch it self-correct if they fall short", elem_classes=["sample-caption"])
            
            # Hidden column to be rendered when the user collapses all settings.
            with gr.Column(scale=1, min_width=100, visible=False) as hidden_settings_column:
                show_settings = gr.Button(value="< Expand", size="sm")
            
            # Right column to display all relevant settings
            with gr.Column(scale=12, min_width=350) as settings_column:
                with gr.Tabs(selected=0) as settings_tabs:

                    with gr.TabItem("Quickstart", id=0) as instructions_tab:

                        gr.Markdown(
                            """
                            ##### What makes this RAG *agentic*?
                            Every answer is **routed** (documents vs. web search), **graded** for relevance,
                            and **verified** for groundedness and usefulness — and the agent **self-corrects**
                            when a check fails. Watch it happen live in the chat and in the **Monitor** tab.
                            """
                        )

                        # Diagram of the agentic websearch RAG workflow
                        with gr.Row():
                            agentic_flow = gr.Image("/project/code/chatui/static/agentic-flow.png",
                                                    show_label=False,
                                                    show_download_button=False,
                                                    interactive=False)
                        gr.Markdown(
                            "The workflow above: a router sends each question to the vector database or web search. "
                            "Retrieved chunks are graded for relevance (irrelevant ones trigger a web-search fallback), "
                            "and every draft answer must pass a groundedness check and an answer check before you see it — "
                            "otherwise the agent regenerates or retries with fresh context.",
                            elem_classes=["diagram-caption"]
                        )

                        with gr.Accordion("Step 1 — Check your setup", open=True):
                            gr.Markdown(
                                """
                                * The status bar above the chat shows whether your ``NVIDIA_API_KEY`` and ``TAVILY_API_KEY``
                                  are configured and what is in your context.
                                * If a key is missing, set it in AI Workbench under **Project Container → Variables**,
                                  then restart this Chat app.
                                """
                            )

                        with gr.Accordion("Step 2 — Ask a sample question and watch the agent work", open=False):
                            gr.Markdown(
                                """
                                * Click a sample under the chat box. Each one is labeled with the agent path it exercises.
                                * While the agent works, the pending chat bubble shows each step live: routing, retrieval,
                                  grading, generation, and verification.
                                * With an empty context, document questions get graded as irrelevant and **fall back to web
                                  search** — that fallback is the agent self-correcting.
                                """
                            )

                        with gr.Accordion("Step 3 — Add the sample docs and see answers improve", open=False):
                            gr.Markdown(
                                """
                                * Open the **Documents** tab and click **Add to Context** under the sample webpage list
                                  (the NVIDIA AI Workbench documentation).
                                * Re-ask the same sample question: the agent now routes to the vector database, keeps the
                                  relevant chunks, and cites its sources under the answer.
                                """
                            )

                        with gr.Accordion("Going further — customize models, prompts, and data", open=False):
                            gr.Markdown(
                                """
                                * **Documents**: clear the context and add your own webpages or files (.pdf, .txt, .csv, .md).
                                * **Models**: pick a preset, change individual component models, or point components at a
                                  self-hosted endpoint. Update the **Router prompt** topics to match your own documents.
                                * **Monitor**: the Actions Console narrates everything; the Response Trace summarizes the last run.
                                """
                            )


                    # Settings for each component model of the agentic workflow
                    with gr.TabItem("Models", id=1) as agent_settings:
                            gr.Markdown(
                                        """
                                        ##### Use the Models tab to configure the agent's model components
                                        - Pick a **preset** to set every component at once, or
                                        - Click a component below (e.g. Router) for fine-grained control: choose an API model,
                                          point at a [self-hosted endpoint](https://github.com/NVIDIA/workbench-example-agentic-rag/blob/main/agentic-rag-docs/self-host.md),
                                          or customize the component's prompt
                                        """
                            )
                            model_preset = gr.Radio(
                                choices=[PRESET_FAST, PRESET_BALANCED, PRESET_QUALITY],
                                value=PRESET_BALANCED,
                                label="Model preset",
                                info="Sets the API endpoint model for all five components below.",
                            )
                            gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                                    
                            ########################
                            ##### ROUTER MODEL #####
                            ########################
                            router_btn = gr.Button("Router", size="sm")
                            with gr.Group(visible=False) as group_router:
                                with gr.Tabs(selected=0) as router_tabs:
                                    with gr.TabItem("API Endpoints", id=0) as router_api:
                                        router_mode_banner = gr.Markdown(value="💻 **Using API Endpoint**", elem_classes=["mode-banner"])
                                        model_router = gr.Dropdown(model_list, 
                                                                value=SUPER,
                                                                label="Select a Model",
                                                                elem_id="rag-inputs", 
                                                                interactive=True)
                                        
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as router_nim:
                                        # with gr.Row():
                                        #     nim_router_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_router_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_router_ip = gr.Textbox(
                                                value = "agentic-rag-local-nim-1",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_router_port = gr.Textbox(
                                                placeholder="8000",
                                                label=HOST_PORT,
                                                info="Optional, (default: 8000)",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_router_id = gr.Textbox(
                                            placeholder = "meta/llama-3.1-8b-instruct",
                                            label=HOST_MODEL,
                                            info="If none specified, defaults to: meta/llama-3.1-8b-instruct",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )
                                        # nim_router_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_router_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as router_hide:
                                        gr.Markdown("")

                                with gr.Accordion("Configure the Router Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_router:
                                    prompt_router = gr.Textbox(value=prompts_nemotron.router_prompt,
                                                            lines=12,
                                                            show_label=False,
                                                            interactive=True)
        
                            ##################################
                            ##### RETRIEVAL GRADER MODEL #####
                            ##################################
                            retrieval_btn = gr.Button("Retrieval Grader", size="sm")
                            with gr.Group(visible=False) as group_retrieval:
                                with gr.Tabs(selected=0) as retrieval_tabs:
                                    retrieval_mode_banner = gr.Markdown(value="💻 **Using API Endpoint**", elem_classes=["mode-banner"])

                                    with gr.TabItem("API Endpoints", id=0) as retrieval_api:
                                        model_retrieval = gr.Dropdown(model_list, 
                                                                            value=SUPER,
                                                                            label="Select a Model",
                                                                            elem_id="rag-inputs", 
                                                                            interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as retrieval_nim:
                                        # with gr.Row():
                                        #     nim_retrieval_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_retrieval_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_retrieval_ip = gr.Textbox(
                                                value = "agentic-rag-local-nim-1",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_retrieval_port = gr.Textbox(
                                                placeholder="8000",
                                                label=HOST_PORT,
                                                info="Optional, (default: 8000)",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_retrieval_id = gr.Textbox(
                                            placeholder = "meta/llama-3.1-8b-instruct",
                                            label=HOST_MODEL,
                                            info="If none specified, defaults to: meta/llama-3.1-8b-instruct",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )                                        
                                        # nim_retrieval_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_retrieval_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as retrieval_hide:
                                        gr.Markdown("")
                                
                                with gr.Accordion("Configure the Retrieval Grader Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_retrieval:
                                    prompt_retrieval = gr.Textbox(value=prompts_nemotron.retrieval_prompt,
                                                                        lines=21,
                                                                        show_label=False,
                                                                        interactive=True)
        
                            ###########################
                            ##### GENERATOR MODEL #####
                            ###########################
                            generator_btn = gr.Button("Generator", size="sm")
                            with gr.Group(visible=False) as group_generator:
                                with gr.Tabs(selected=0) as generator_tabs:
                                    generator_mode_banner = gr.Markdown(value="💻 **Using API Endpoint**", elem_classes=["mode-banner"])
                                    with gr.TabItem("API Endpoints", id=0) as generator_api:
                                        model_generator = gr.Dropdown(model_list, 
                                                                    value=SUPER,
                                                                    label="Select a Model",
                                                                    elem_id="rag-inputs", 
                                                                    interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as generator_nim:
                                        # with gr.Row():
                                        #     nim_generator_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_generator_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_generator_ip = gr.Textbox(
                                                value = "agentic-rag-local-nim-1",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_generator_port = gr.Textbox(
                                                placeholder="8000",
                                                label=HOST_PORT,
                                                info="Optional, (default: 8000)",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_generator_id = gr.Textbox(
                                            placeholder = "meta/llama-3.1-8b-instruct",
                                            label=HOST_MODEL,
                                            info="If none specified, defaults to: meta/llama-3.1-8b-instruct",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )
                                        # nim_generator_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_generator_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as generator_hide:
                                        gr.Markdown("")
                                
                                with gr.Accordion("Configure the Generator Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_generator:
                                    prompt_generator = gr.Textbox(value=prompts_nemotron.generator_prompt,
                                                            lines=15,
                                                            show_label=False,
                                                            interactive=True)
        
                            ######################################
                            ##### HALLUCINATION GRADER MODEL #####
                            ######################################
                            hallucination_btn = gr.Button("Hallucination Grader", size="sm")
                            with gr.Group(visible=False) as group_hallucination:
                                with gr.Tabs(selected=0) as hallucination_tabs:
                                    hallucination_mode_banner = gr.Markdown(value="💻 **Using API Endpoint**", elem_classes=["mode-banner"])
                                    with gr.TabItem("API Endpoints", id=0) as hallucination_api:
                                        model_hallucination = gr.Dropdown(model_list, 
                                                                                value=SUPER,
                                                                                label="Select a Model",
                                                                                elem_id="rag-inputs", 
                                                                                interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as hallucination_nim:
                                        # with gr.Row():
                                        #     nim_hallucination_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_hallucination_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_hallucination_ip = gr.Textbox(
                                                value = "agentic-rag-local-nim-1",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_hallucination_port = gr.Textbox(
                                                placeholder="8000",
                                                label=HOST_PORT,
                                                info="Optional, (default: 8000)",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_hallucination_id = gr.Textbox(
                                            placeholder = "meta/llama-3.1-8b-instruct",
                                            label=HOST_MODEL,
                                            info="If none specified, defaults to: meta/llama-3.1-8b-instruct",
                                            elem_id="rag-inputs",
                                            interactive=True
                                        )
                                        # nim_hallucination_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_hallucination_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as hallucination_hide:
                                        gr.Markdown("")
                                
                                with gr.Accordion("Configure the Hallucination Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_hallucination:
                                    prompt_hallucination = gr.Textbox(value=prompts_nemotron.hallucination_prompt,
                                                                            lines=17,
                                                                            show_label=False,
                                                                            interactive=True)
        
                            ###############################
                            ##### ANSWER GRADER MODEL #####
                            ###############################
                            answer_btn = gr.Button("Answer Grader", size="sm")
                            with gr.Group(visible=False) as group_answer:
                                with gr.Tabs(selected=0) as answer_tabs:
                                    answer_mode_banner = gr.Markdown(value="💻 **Using API Endpoint**", elem_classes=["mode-banner"])
                                    with gr.TabItem("API Endpoints", id=0) as answer_api:
                                        model_answer = gr.Dropdown(model_list, 
                                                                        value=SUPER,
                                                                        elem_id="rag-inputs",
                                                                        label="Select a Model",
                                                                        interactive=True)
                                    with gr.TabItem(SELF_HOSTED_TAB_NAME, id=1) as answer_nim:
                                        # with gr.Row():
                                        #     nim_answer_gpu_type = gr.Dropdown(
                                        #         choices=gpu_compatibility.get_gpu_types(),
                                        #         label="GPU Type",
                                        #         info="Select your GPU type",
                                        #         elem_id="rag-inputs",
                                        #         scale=2
                                        #     )
                                        #     nim_answer_gpu_count = gr.Dropdown(
                                        #         choices=[],
                                        #         label="Number of GPUs",
                                        #         info="Select number of GPUs",
                                        #         elem_id="rag-inputs",
                                        #         scale=1,
                                        #         interactive=False
                                        #     )
                                        
                                        with gr.Row():
                                            nim_answer_ip = gr.Textbox(
                                                value = "agentic-rag-local-nim-1",
                                                label=HOST_NAME,
                                                info="Local microservice OR IP address running a remote microservice",
                                                elem_id="rag-inputs",
                                                scale=2
                                            )
                                            nim_answer_port = gr.Textbox(
                                                placeholder="8000",
                                                label=HOST_PORT,
                                                info="Optional, (default: 8000)",
                                                elem_id="rag-inputs",
                                                scale=1
                                            )
                                        nim_answer_id = gr.Textbox(
                                            placeholder = "meta/llama-3.1-8b-instruct",
                                            label=HOST_MODEL,
                                            info="If none specified, defaults to: meta/llama-3.1-8b-instruct",
                                            elem_id="rag-inputs",
                                            interactive=True
                                            )   

                                        # nim_answer_id = gr.Dropdown(
                                        #     choices=[],
                                        #     label="Model running in microservice",
                                        #     info="Select a compatible model for your GPU configuration",
                                        #     elem_id="rag-inputs",
                                        #     interactive=False
                                        # )

                                        # Add warning box for compatibility issues
                                        nim_answer_warning = gr.Markdown(visible=False, value="")

                                    with gr.TabItem("Hide", id=2) as answer_hide:
                                        gr.Markdown("")
                                        
                                with gr.Accordion("Configure the Answer Prompt", 
                                                elem_id="rag-inputs", open=False) as accordion_answer:
                                    prompt_answer = gr.Textbox(value=prompts_nemotron.answer_prompt,
                                                                    lines=17,
                                                                    show_label=False,
                                                                    interactive=True)
                        
                    # Third tab item is for uploading to and clearing the vector database
                    with gr.TabItem("Documents", id=2) as document_settings:
                        gr.Markdown(
                            """
                            ##### Use the Documents tab to manage the RAG context
                            - Webpages: Enter URLs of webpages for the context
                            - Files: Use files (.pdf, .txt, .csv, .md) for the context
                            - Add to Context: Add documents to the context. Context is stored until you clear it.
                            - Clear Context: Resets the context to empty
                            """
                            )
                        with gr.Accordion("What's in the context right now", open=True):
                            context_panel = gr.Markdown("⏳ Loading the context summary…")
                            context_refresh_btn = gr.Button("Refresh summary", size="sm")
                        gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                        with gr.Tabs(selected=0) as document_tabs:
                            with gr.TabItem("Webpages", id=0) as url_tab:
                                url_docs = gr.Textbox(value=EXAMPLE_LINKS,
                                                      lines=EXAMPLE_LINKS_LEN, 
                                                      info="Enter a list of URLs, one per line", 
                                                      show_label=False, 
                                                      interactive=True)
                            
                                with gr.Row():
                                    url_docs_upload = gr.Button(value="Add to Context")
                                    url_docs_clear = gr.Button(value="Clear Context")

                            with gr.TabItem("Files", id=1) as pdf_tab:
                                docs_upload = gr.File(interactive=True, 
                                                          show_label=False, 
                                                          file_types=[".pdf", ".txt", ".csv", ".md"], 
                                                          file_count="multiple")
                                docs_clear = gr.Button(value="Clear Context")
    
                    # Fourth tab item is for the actions output console.
                    with gr.TabItem("Monitor", id=3) as console_settings:
                        gr.Markdown(
                            """
                            ##### Use the Monitor tab to see the agent in action
                            - Actions Console: a live narration of every routing, grading, and generation step
                            - Response Trace: a structured summary of the latest response (steps, self-corrections, sources)
                            """
                            )
                        gr.HTML('<hr style="border:1px solid #ccc; margin: 10px 0;">')
                        with gr.Tabs(selected=0) as console_tabs:
                            with gr.TabItem("Actions Console", id=0) as actions_tab:
                                logs = gr.Textbox(show_label=False, lines=18, max_lines=18, interactive=False)
                            with gr.TabItem("Response Trace", id=1) as trace_tab:
                                gr.Markdown("Updates live while a query runs. Fields: the question, run status, "
                                            "each agent step, the self-correction count, and the sources used.")
                                actions = gr.JSON(
                                    scale=1,
                                    show_label=False,
                                    visible=True,
                                    elem_id="contextbox",
                                )
                    
                    # Fifth tab item is for collapsing the entire settings pane for readability. 
                    with gr.TabItem("Hide All Settings", id=4) as hide_all_settings:
                        gr.Markdown("")

        page.load(logger.read_logs, None, logs, every=1)
        page.load(_refresh_context_displays, None, [status_strip, context_panel])

        """ These helper functions hide all settings when collapsed, and displays all settings when expanded. """

        def _toggle_hide_all_settings():
            return {
                settings_column: gr.update(visible=False),
                hidden_settings_column: gr.update(visible=True),
            }

        def _toggle_show_all_settings():
            return {
                settings_column: gr.update(visible=True),
                settings_tabs: gr.update(selected=0),
                hidden_settings_column: gr.update(visible=False),
            }

        hide_all_settings.select(_toggle_hide_all_settings, None, [settings_column, hidden_settings_column])
        show_settings.click(_toggle_show_all_settings, None, [settings_column, settings_tabs, hidden_settings_column])

        """ These helper functions hide the expanded component model settings when the Hide tab is clicked. """
        
        def _toggle_hide_router():
            return {
                group_router: gr.update(visible=False),
                router_tabs: gr.update(selected=0),
                router_btn: gr.update(visible=True),
            }

        def _toggle_hide_retrieval():
            return {
                group_retrieval: gr.update(visible=False),
                retrieval_tabs: gr.update(selected=0),
                retrieval_btn: gr.update(visible=True),
            }

        def _toggle_hide_generator():
            return {
                group_generator: gr.update(visible=False),
                generator_tabs: gr.update(selected=0),
                generator_btn: gr.update(visible=True),
            }

        def _toggle_hide_hallucination():
            return {
                group_hallucination: gr.update(visible=False),
                hallucination_tabs: gr.update(selected=0),
                hallucination_btn: gr.update(visible=True),
            }

        def _toggle_hide_answer():
            return {
                group_answer: gr.update(visible=False),
                answer_tabs: gr.update(selected=0),
                answer_btn: gr.update(visible=True),
            }

        router_hide.select(_toggle_hide_router, None, [group_router, router_tabs, router_btn])
        retrieval_hide.select(_toggle_hide_retrieval, None, [group_retrieval, retrieval_tabs, retrieval_btn])
        generator_hide.select(_toggle_hide_generator, None, [group_generator, generator_tabs, generator_btn])
        hallucination_hide.select(_toggle_hide_hallucination, None, [group_hallucination, hallucination_tabs, hallucination_btn])
        answer_hide.select(_toggle_hide_answer, None, [group_answer, answer_tabs, answer_btn])

        """ These helper functions set state and prompts when either the NIM or API Endpoint tabs are selected. """
        
        def _update_gpu_counts(component: str, gpu_type: str):
            """Update the available GPU counts for selected GPU type."""
            counts = gpu_compatibility.get_supported_gpu_counts(gpu_type)
            components = {
                "router": [nim_router_gpu_count, nim_router_id, nim_router_warning],
                "retrieval": [nim_retrieval_gpu_count, nim_retrieval_id, nim_retrieval_warning],
                "generator": [nim_generator_gpu_count, nim_generator_id, nim_generator_warning],
                "hallucination": [nim_hallucination_gpu_count, nim_hallucination_id, nim_hallucination_warning],
                "answer": [nim_answer_gpu_count, nim_answer_id, nim_answer_warning]
            }
            return {
                components[component][0]: gr.update(choices=counts, value=None, interactive=True),
                components[component][1]: gr.update(choices=[], value=None, interactive=False),
                components[component][2]: gr.update(visible=False, value="")
            }
        
        def _update_compatible_models(component: str, gpu_type: str, num_gpus: str):
            """Update the compatible models list based on GPU configuration."""
            if not gpu_type or not num_gpus:
                components = {
                    "router": [nim_router_id, nim_router_warning],
                    "retrieval": [nim_retrieval_id, nim_retrieval_warning],
                    "generator": [nim_generator_id, nim_generator_warning],
                    "hallucination": [nim_hallucination_id, nim_hallucination_warning],
                    "answer": [nim_answer_id, nim_answer_warning]
                }
                return {
                    components[component][0]: gr.update(choices=[], value=None, interactive=False),
                    components[component][1]: gr.update(visible=False, value="")
                }
            
            compatibility = gpu_compatibility.get_compatible_models(gpu_type, num_gpus)
            
            if compatibility["warning_message"]:
                components = {
                    "router": [nim_router_id, nim_router_warning],
                    "retrieval": [nim_retrieval_id, nim_retrieval_warning],
                    "generator": [nim_generator_id, nim_generator_warning],
                    "hallucination": [nim_hallucination_id, nim_hallucination_warning],
                    "answer": [nim_answer_id, nim_answer_warning]
                }
                return {
                    components[component][0]: gr.update(choices=[], value=None, interactive=False),
                    components[component][1]: gr.update(visible=True, value=f"⚠️ {compatibility['warning_message']}")
                }
            
            components = {
                "router": [nim_router_id, nim_router_warning],
                "retrieval": [nim_retrieval_id, nim_retrieval_warning],
                "generator": [nim_generator_id, nim_generator_warning],
                "hallucination": [nim_hallucination_id, nim_hallucination_warning],
                "answer": [nim_answer_id, nim_answer_warning]
            }
            return {
                components[component][0]: gr.update(
                    choices=compatibility["compatible_models"],
                    value=compatibility["compatible_models"][0] if compatibility["compatible_models"] else None,
                    interactive=True
                ),
                components[component][1]: gr.update(visible=False, value="")
            }

        # Add the event handlers for all components
        # nim_router_gpu_type.change(lambda x: _update_gpu_counts("router", x), nim_router_gpu_type, 
        #                          [nim_router_gpu_count, nim_router_id, nim_router_warning])
        # nim_router_gpu_count.change(lambda x, y: _update_compatible_models("router", x, y), 
        #                           [nim_router_gpu_type, nim_router_gpu_count], 
        #                           [nim_router_id, nim_router_warning])

        # nim_retrieval_gpu_type.change(lambda x: _update_gpu_counts("retrieval", x), nim_retrieval_gpu_type, 
        #                             [nim_retrieval_gpu_count, nim_retrieval_id, nim_retrieval_warning])
        # nim_retrieval_gpu_count.change(lambda x, y: _update_compatible_models("retrieval", x, y), 
        #                              [nim_retrieval_gpu_type, nim_retrieval_gpu_count], 
        #                              [nim_retrieval_id, nim_retrieval_warning])

        # nim_generator_gpu_type.change(lambda x: _update_gpu_counts("generator", x), nim_generator_gpu_type, 
        #                             [nim_generator_gpu_count, nim_generator_id, nim_generator_warning])
        # nim_generator_gpu_count.change(lambda x, y: _update_compatible_models("generator", x, y), 
        #                              [nim_generator_gpu_type, nim_generator_gpu_count], 
        #                              [nim_generator_id, nim_generator_warning])

        # nim_hallucination_gpu_type.change(lambda x: _update_gpu_counts("hallucination", x), nim_hallucination_gpu_type, 
        #                                 [nim_hallucination_gpu_count, nim_hallucination_id, nim_hallucination_warning])
        # nim_hallucination_gpu_count.change(lambda x, y: _update_compatible_models("hallucination", x, y), 
        #                                  [nim_hallucination_gpu_type, nim_hallucination_gpu_count], 
        #                                  [nim_hallucination_id, nim_hallucination_warning])

        # nim_answer_gpu_type.change(lambda x: _update_gpu_counts("answer", x), nim_answer_gpu_type, 
        #                          [nim_answer_gpu_count, nim_answer_id, nim_answer_warning])
        # nim_answer_gpu_count.change(lambda x, y: _update_compatible_models("answer", x, y), 
        #                           [nim_answer_gpu_type, nim_answer_gpu_count], 
        #                           [nim_answer_id, nim_answer_warning])

        """ This helper applies a model preset to all five component dropdowns at once. """

        def _apply_model_preset(preset: str):
            router_m, retrieval_m, generator_m, hallucination_m, answer_m = MODEL_PRESETS[preset]
            return (gr.update(value=router_m),
                    gr.update(value=retrieval_m),
                    gr.update(value=generator_m),
                    gr.update(value=hallucination_m),
                    gr.update(value=answer_m))

        model_preset.change(_apply_model_preset,
                            [model_preset],
                            [model_router, model_retrieval, model_generator, model_hallucination, model_answer])

        """ These helper functions track the API Endpoint selected and regenerate the prompt accordingly.
        All Nemotron 3 models (Nano, Super, Ultra) share the same prompt structure, so switching models
        resets the prompt to the same shared default. """

        def _toggle_model_router(selected_model: str):
            return gr.update(value=prompts_nemotron.router_prompt)
        
        def _toggle_model_retrieval(selected_model: str):
            return gr.update(value=prompts_nemotron.retrieval_prompt)

        def _toggle_model_generator(selected_model: str):
            return gr.update(value=prompts_nemotron.generator_prompt)
            
        def _toggle_model_hallucination(selected_model: str):
            return gr.update(value=prompts_nemotron.hallucination_prompt)
            
        def _toggle_model_answer(selected_model: str):
            return gr.update(value=prompts_nemotron.answer_prompt)

        # Update default prompts when an API endpoint model is selected from the dropdown
        # (This applies only to the "API Endpoints" tab — not to self-hosted NIM configurations)

        model_router.change(_toggle_model_router, [model_router], [prompt_router])
        model_retrieval.change(_toggle_model_retrieval, [model_retrieval], [prompt_retrieval])
        model_generator.change(_toggle_model_generator, [model_generator], [prompt_generator])
        model_hallucination.change(_toggle_model_hallucination, [model_hallucination], [prompt_hallucination])
        model_answer.change(_toggle_model_answer, [model_answer], [prompt_answer])

        # Toggle between NIM and API mode by setting `*_use_nim` state based on selected tab
        # - Selecting "API Endpoints" sets use_nim = False (use hosted model)
        # - Selecting "Self-Hosted Endpoint" sets use_nim = True (use local NIM container)
        
        # router eventhandlers
        router_api.select(lambda: (False,), [], [router_use_nim])
        router_nim.select(lambda: (True,), [], [router_use_nim])

        router_api.select(lambda: "💻 **Using API Endpoint**", [], [router_mode_banner])
        router_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [router_mode_banner])

        # retrieval eventhandlers   
        retrieval_api.select(lambda: (False,), [], [retrieval_use_nim])
        retrieval_nim.select(lambda: (True,), [], [retrieval_use_nim])

        retrieval_api.select(lambda: "💻 **Using API Endpoint**", [], [retrieval_mode_banner])
        retrieval_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [retrieval_mode_banner])

        # generator eventhandlers
        generator_api.select(lambda: (False,), [], [generator_use_nim])
        generator_nim.select(lambda: (True,), [], [generator_use_nim])

        generator_api.select(lambda: "💻 **Using API Endpoint**", [], [generator_mode_banner])
        generator_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [generator_mode_banner])

        # hallucination eventhandlers
        hallucination_api.select(lambda: (False,), [], [hallucination_use_nim])
        hallucination_nim.select(lambda: (True,), [], [hallucination_use_nim])

        hallucination_api.select(lambda: "💻 **Using API Endpoint**", [], [hallucination_mode_banner])
        hallucination_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [hallucination_mode_banner])

        # answer eventhandlers
        answer_api.select(lambda: (False,), [], [answer_use_nim])
        answer_nim.select(lambda: (True,), [], [answer_use_nim])

        answer_api.select(lambda: "💻 **Using API Endpoint**", [], [answer_mode_banner])
        answer_nim.select(lambda: "🛠️ **Using Self-Hosted Endpoint**", [], [answer_mode_banner])
        
        """ These helper functions upload and clear the documents and webpages to/from the ChromaDB.
        Each one also refreshes the status strip and the context panel so the UI always reflects
        what is actually in the vector database. Detailed per-document progress is narrated in
        Monitor → Actions Console. """

        def _upload_documents_files(files, progress=gr.Progress()):
            progress(0.1, desc="Reading files")
            progress(0.3, desc="Loading and embedding files (details in Monitor → Actions Console)")
            database.upload_files(files)
            progress(0.9, desc="Updating context summary")
            status, panel = _refresh_context_displays()
            return {
                url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
                status_strip: gr.update(value=status),
                context_panel: gr.update(value=panel),
            }

        def _upload_documents(docs: str, progress=gr.Progress()):
            progress(0.1, desc="Reading URL list")
            docs_list = docs.splitlines()
            progress(0.3, desc="Loading and embedding webpages (details in Monitor → Actions Console)")
            vectorstore = database.upload(docs_list)
            progress(0.9, desc="Updating context summary")
            status, panel = _refresh_context_displays()
            if vectorstore is None:
                has_context = database.get_context_summary()[0] > 0
                return {
                    url_docs_upload: gr.update(value="No valid URLs — try again", variant="secondary", interactive=True),
                    url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=has_context),
                    docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=has_context),
                    agentic_flow: gr.update(visible=True),
                    status_strip: gr.update(value=status),
                    context_panel: gr.update(value=panel),
                }
            return {
                url_docs_upload: gr.update(value="Context Created", variant="primary", interactive=False),
                url_docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                docs_clear: gr.update(value="Clear Context", variant="secondary", interactive=True),
                agentic_flow: gr.update(visible=True),
                status_strip: gr.update(value=status),
                context_panel: gr.update(value=panel),
            }

        def _clear_documents(progress=gr.Progress()):
            progress(0.3, desc="Clearing the context")
            database._clear()
            progress(0.8, desc="Updating context summary")
            status, panel = _refresh_context_displays()
            return {
                url_docs_upload: gr.update(value="Add to Context", variant="secondary", interactive=True),
                url_docs_clear: gr.update(value="Context Cleared", variant="primary", interactive=False),
                docs_upload: gr.update(value=None),
                docs_clear: gr.update(value="Context Cleared", variant="primary", interactive=False),
                agentic_flow: gr.update(visible=True),
                status_strip: gr.update(value=status),
                context_panel: gr.update(value=panel),
            }

        url_docs_upload.click(_upload_documents, [url_docs],
                              [url_docs_upload, url_docs_clear, docs_clear, agentic_flow, status_strip, context_panel])
        url_docs_clear.click(_clear_documents, [],
                             [url_docs_upload, url_docs_clear, docs_upload, docs_clear, agentic_flow, status_strip, context_panel])
        docs_upload.upload(_upload_documents_files, [docs_upload],
                           [url_docs_clear, docs_clear, agentic_flow, status_strip, context_panel])
        docs_clear.click(_clear_documents, [],
                         [url_docs_upload, url_docs_clear, docs_upload, docs_clear, agentic_flow, status_strip, context_panel])
        context_refresh_btn.click(_refresh_context_displays, None, [status_strip, context_panel])

        """ These helper functions set state and prompts when either the NIM or API Endpoint tabs are selected. """
        
        def _toggle_model_tab():
            return {
                group_router: gr.update(visible=False),
                group_retrieval: gr.update(visible=False),
                group_generator: gr.update(visible=False),
                group_hallucination: gr.update(visible=False),
                group_answer: gr.update(visible=False),
                router_btn: gr.update(visible=True),
                retrieval_btn: gr.update(visible=True),
                generator_btn: gr.update(visible=True),
                hallucination_btn: gr.update(visible=True),
                answer_btn: gr.update(visible=True),
            }
        
        agent_settings.select(_toggle_model_tab, [], [group_router,
                                                      group_retrieval,
                                                      group_generator,
                                                      group_hallucination,
                                                      group_answer,
                                                      router_btn,
                                                      retrieval_btn,
                                                      generator_btn,
                                                      hallucination_btn,
                                                      answer_btn])

        """ This helper function ensures only one component model settings are expanded at a time when selected. """

        def _toggle_model(btn: str):
            if btn == "Router":
                group_visible = [True, False, False, False, False]
                button_visible = [False, True, True, True, True]
            elif btn == "Retrieval Grader":
                group_visible = [False, True, False, False, False]
                button_visible = [True, False, True, True, True]
            elif btn == "Generator":
                group_visible = [False, False, True, False, False]
                button_visible = [True, True, False, True, True]
            elif btn == "Hallucination Grader":
                group_visible = [False, False, False, True, False]
                button_visible = [True, True, True, False, True]
            elif btn == "Answer Grader":
                group_visible = [False, False, False, False, True]
                button_visible = [True, True, True, True, False]
            return {
                group_router: gr.update(visible=group_visible[0]),
                group_retrieval: gr.update(visible=group_visible[1]),
                group_generator: gr.update(visible=group_visible[2]),
                group_hallucination: gr.update(visible=group_visible[3]),
                group_answer: gr.update(visible=group_visible[4]),
                router_btn: gr.update(visible=button_visible[0]),
                retrieval_btn: gr.update(visible=button_visible[1]),
                generator_btn: gr.update(visible=button_visible[2]),
                hallucination_btn: gr.update(visible=button_visible[3]),
                answer_btn: gr.update(visible=button_visible[4]),
            }

        router_btn.click(_toggle_model, [router_btn], [group_router,
                                                       group_retrieval,
                                                       group_generator,
                                                       group_hallucination,
                                                       group_answer,
                                                       router_btn,
                                                       retrieval_btn,
                                                       generator_btn,
                                                       hallucination_btn,
                                                       answer_btn])
        
        retrieval_btn.click(_toggle_model, [retrieval_btn], [group_router,
                                                                           group_retrieval,
                                                                           group_generator,
                                                                           group_hallucination,
                                                                           group_answer,
                                                                           router_btn,
                                                                           retrieval_btn,
                                                                           generator_btn,
                                                                           hallucination_btn,
                                                                           answer_btn])
        
        generator_btn.click(_toggle_model, [generator_btn], [group_router,
                                                             group_retrieval,
                                                             group_generator,
                                                             group_hallucination,
                                                             group_answer,
                                                             router_btn,
                                                             retrieval_btn,
                                                             generator_btn,
                                                             hallucination_btn,
                                                             answer_btn])
        
        hallucination_btn.click(_toggle_model, [hallucination_btn], [group_router,
                                                                                   group_retrieval,
                                                                                   group_generator,
                                                                                   group_hallucination,
                                                                                   group_answer,
                                                                                   router_btn,
                                                                                   retrieval_btn,
                                                                                   generator_btn,
                                                                                   hallucination_btn,
                                                                                   answer_btn])
        
        answer_btn.click(_toggle_model, [answer_btn], [group_router,
                                                                     group_retrieval,
                                                                     group_generator,
                                                                     group_hallucination,
                                                                     group_answer,
                                                                     router_btn,
                                                                     retrieval_btn,
                                                                     generator_btn,
                                                                     hallucination_btn,
                                                                     answer_btn])

        """ This helper function builds out the submission function call when a user submits a query. """
        
        _my_build_stream = functools.partial(_stream_predict, client, app)

        # Submit a query. Sample buttons pass their own label as the query text, so the
        # input list is identical for every trigger and defined exactly once.
        stream_inputs = [model_generator,
                         model_router,
                         model_retrieval,
                         model_hallucination,
                         model_answer,
                         prompt_generator,
                         prompt_router,
                         prompt_retrieval,
                         prompt_hallucination,
                         prompt_answer,
                         router_use_nim,
                         retrieval_use_nim,
                         generator_use_nim,
                         hallucination_use_nim,
                         answer_use_nim,
                         nim_generator_ip,
                         nim_router_ip,
                         nim_retrieval_ip,
                         nim_hallucination_ip,
                         nim_answer_ip,
                         nim_generator_port,
                         nim_router_port,
                         nim_retrieval_port,
                         nim_hallucination_port,
                         nim_answer_port,
                         nim_generator_id,
                         nim_router_id,
                         nim_retrieval_id,
                         nim_hallucination_id,
                         nim_answer_id,
                         chatbot]
        stream_outputs = [msg, chatbot, actions]

        for sample_query in (sample_query_1, sample_query_2, sample_query_3, sample_query_4):
            sample_query.click(_my_build_stream, [sample_query] + stream_inputs, stream_outputs)

        msg.submit(_my_build_stream, [msg] + stream_inputs, stream_outputs)

    page.queue()
    return page

""" This helper function verifies that a user query is nonempty. """

def valid_input(query: str):
    return False if query.isspace() or query is None or query == "" or query == '' else True


""" This helper function provides error outputs for the query. """
def _get_query_error_message(e: Exception) -> str:
    if isinstance(e, GraphRecursionError):
        err = QUERY_ERROR_MESSAGES["GraphRecursionError"]
    elif isinstance(e, HTTPError):
        if e.response is not None and e.response.status_code == 401:
            err = QUERY_ERROR_MESSAGES["AuthenticationError"]
        else:
            err = QUERY_ERROR_MESSAGES["HTTPError"]
    elif isinstance(e, TavilyAPIError):
        err = QUERY_ERROR_MESSAGES["TavilyAPIError"]
    elif isinstance(e, OutputParserException):
        err = QUERY_ERROR_MESSAGES["OutputParserError"]
    else:
        err = QUERY_ERROR_MESSAGES["Unknown"]

    return f"{err['title']}\n\n{err['body']}"




""" This helper function executes and generates a response to the user query. """
def _stream_predict(
    client: chat_client.ChatClient,
    app, 
    question: str,
    model_generator: str,
    model_router: str,
    model_retrieval: str,
    model_hallucination: str,
    model_answer: str,
    prompt_generator: str,
    prompt_router: str,
    prompt_retrieval: str,
    prompt_hallucination: str,
    prompt_answer: str,
    router_use_nim: bool,
    retrieval_use_nim: bool,
    generator_use_nim: bool,
    hallucination_use_nim: bool,
    answer_use_nim: bool,
    nim_generator_ip: str,
    nim_router_ip: str,
    nim_retrieval_ip: str,
    nim_hallucination_ip: str,
    nim_answer_ip: str,
    nim_generator_port: str,
    nim_router_port: str,
    nim_retrieval_port: str,
    nim_hallucination_port: str,
    nim_answer_port: str,
    nim_generator_id: str,
    nim_router_id: str,
    nim_retrieval_id: str,
    nim_hallucination_id: str,
    nim_answer_id: str,
    chat_history: List[Tuple[str, str]],
) -> Any:

    inputs = {"question": question, 
              "generator_model_id": model_generator, 
              "router_model_id": model_router, 
              "retrieval_model_id": model_retrieval, 
              "hallucination_model_id": model_hallucination, 
              "answer_model_id": model_answer, 
              "prompt_generator": prompt_generator, 
              "prompt_router": prompt_router, 
              "prompt_retrieval": prompt_retrieval, 
              "prompt_hallucination": prompt_hallucination, 
              "prompt_answer": prompt_answer, 
              "router_use_nim": router_use_nim, 
              "retrieval_use_nim": retrieval_use_nim, 
              "generator_use_nim": generator_use_nim, 
              "hallucination_use_nim": hallucination_use_nim, 
              "nim_generator_ip": nim_generator_ip,
              "nim_router_ip": nim_router_ip,
              "nim_retrieval_ip": nim_retrieval_ip,
              "nim_hallucination_ip": nim_hallucination_ip,
              "nim_answer_ip": nim_answer_ip,
              "nim_generator_port": nim_generator_port,
              "nim_router_port": nim_router_port,
              "nim_retrieval_port": nim_retrieval_port,
              "nim_hallucination_port": nim_hallucination_port,
              "nim_answer_port": nim_answer_port,
              "nim_generator_id": nim_generator_id,
              "nim_router_id": nim_router_id,
              "nim_retrieval_id": nim_retrieval_id,
              "nim_hallucination_id": nim_hallucination_id,
              "nim_answer_id": nim_answer_id,
              "answer_use_nim": answer_use_nim}
    
    if not valid_input(question):
        yield "", chat_history + [[str(question), "*** ERR: Unable to process query. Query cannot be empty. ***"]], gr.update(show_label=False)
    else:
        print("\n[Query] ────────────────────────────────────────────")
        print(f'[Query] Processing: "{question}"')
        timeline = _new_timeline()
        final_state = None
        try:
            config = RunnableConfig(recursion_limit=RECURSION_LIMIT)
            for output in app.stream(inputs, config=config):
                # Each streamed event is {node_name: state_delta}; narrate it in the
                # pending chat bubble and in the Response Trace as it happens.
                for node, delta in output.items():
                    _record_timeline_event(timeline, node, delta)
                    final_state = delta
                trace = _build_trace(question, timeline)
                yield "", chat_history + [[question, _timeline_md(timeline, working=True)]], gr.update(value=trace)

            if not final_state or "generation" not in final_state:
                raise RuntimeError("The agent finished without producing an answer.")

            _finish_timeline(timeline)
            documents = final_state.get("documents") or []
            answer = _final_answer_md(final_state["generation"], timeline, documents)
            trace = _build_trace(question, timeline, documents=documents, status="response delivered")
            print("[Query] ✓ Response delivered to the chat window")
            yield "", chat_history + [[question, answer]], gr.update(value=trace)

        except Exception as e:
            traceback.print_exc()

            message = _get_query_error_message(e)
            print(f"[Query] ✗ Query failed ({type(e).__name__}) — an explanation was posted to the chat window")

            # Keep whatever progress the agent made visible above the error explanation.
            if timeline["steps"]:
                message = _timeline_md(timeline, working=False) + "\n\n---\n\n" + message
            trace = _build_trace(question, timeline, status=f"failed ({type(e).__name__})")
            yield "", chat_history + [[question, message]], gr.update(value=trace)


_support_matrix_cache = None

def load_gpu_support_matrix() -> Dict:
    global _support_matrix_cache
    if _support_matrix_cache is None:
        matrix_path = os.path.join(os.path.dirname(__file__), '..', '..', 'nim_gpu_support_matrix.json')
        with open(matrix_path, 'r') as f:
            _support_matrix_cache = json.load(f)
    return _support_matrix_cache

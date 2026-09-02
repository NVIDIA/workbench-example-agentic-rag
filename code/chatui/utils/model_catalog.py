import os
from typing import List, Tuple

import requests


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NON_CHAT_ID_PARTS = (
    "audio",
    "embed",
    "guard",
    "moderation",
    "ocr",
    "rerank",
    "reward",
    "safety",
    "speech",
    "tts",
)


def fetch_model_ids() -> List[str]:
    """Return model IDs exposed to the configured NVIDIA API key."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not configured")

    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    response = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json().get("data")
    if not isinstance(data, list):
        raise RuntimeError("NVIDIA model catalog returned no model list")

    return sorted(
        {
            item["id"]
            for item in data
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    )


def fetch_model_choices() -> Tuple[List[str], List[str]]:
    """Return live chat and embedding model choices."""
    model_ids = [
        model_id
        for model_id in fetch_model_ids()
        if model_id.startswith("nvidia/")
    ]
    embedding_models = [
        model_id for model_id in model_ids if "embed" in model_id.lower()
    ]
    chat_models = [
        model_id
        for model_id in model_ids
        if not any(part in model_id.lower() for part in NON_CHAT_ID_PARTS)
    ]
    return chat_models, embedding_models

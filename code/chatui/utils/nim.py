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

from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load.dump import dumps
from pydantic import Field
from typing import List, Mapping, Optional, Any
from chatui.utils import gpu_compatibility
import os

class CustomChatOpenAI(BaseChatModel):
    """ This is a custom built class for using LangChain to chat with custom OpenAI API-compatible endpoints, eg. NIMs. """

    custom_endpoint: str = Field(None, description='Endpoint of remotely running NIM')
    port: Optional[str] = "8000"
    model_name: Optional[str] = "meta/llama3-8b-instruct"
    temperature: Optional[float] = 0.0
    gpu_type: Optional[str] = None
    gpu_count: Optional[str] = None

    def __init__(self, custom_endpoint, port="8000", model_name="meta/llama3-8b-instruct", 
                 gpu_type=None, gpu_count=None, temperature=0.0, **kwargs):
        super().__init__(**kwargs)
        if gpu_type and gpu_count:
            compatibility = gpu_compatibility.get_compatible_models(gpu_type, gpu_count)
            if compatibility["warning_message"]:
                raise ValueError(compatibility["warning_message"])
            if model_name not in compatibility["compatible_models"]:
                raise ValueError(f"Model {model_name} is not compatible with {gpu_type} ({gpu_count} GPUs)")
        self.custom_endpoint = custom_endpoint
        self.port = port
        self.model_name = model_name
        self.temperature = temperature
        self.gpu_type = gpu_type
        self.gpu_count = gpu_count

    @property
    def _llm_type(self) -> str:
        return 'llama'
        
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self._call_custom_endpoint(messages)
        return self._create_chat_result(response)
    
    def _call_custom_endpoint(self, messages, **kwargs):
        import openai
        import json
        
        openai.api_key = os.getenv("OPENAI_API_KEY", "xyz")  # Better API key handling
        openai.base_url = f"http://{self.custom_endpoint}:{self.port}/v1/"
        
        obj = json.loads(dumps(messages))
        
        config = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": obj[0]["kwargs"]["content"]}],
            "temperature": self.temperature,
        }
        
        if self.gpu_type and self.gpu_count:
            config["gpu_config"] = {
                "type": self.gpu_type,
                "count": self.gpu_count
            }
        
        try:
            response = openai.chat.completions.create(**config)
            return response
        except Exception as e:
            if self.gpu_type and self.gpu_count:
                raise ValueError(f"Error with GPU configuration ({self.gpu_type}, {self.gpu_count} GPUs): {str(e)}")
            raise e
    
    def _create_chat_result(self, response):
        from langchain_core.messages import ChatMessage
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        message = ChatMessage(content=response.choices[0].message.content, role="assistant")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

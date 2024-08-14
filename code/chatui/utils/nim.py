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

class CustomChatOpenAI(BaseChatModel):
    """ This is a custom built class for using LangChain to chat with custom OpenAI API-compatible endpoints, eg. NIMs. """

    custom_endpoint: str = Field(None, description='Endpoint of remotely running NIM')
    port: Optional[str] = "8000"
    model_name: Optional[str] = "meta/llama3-8b-instruct"
    temperature: Optional[float] = 0.0

    def __init__(self, custom_endpoint, port="8000", model_name="meta/llama3-8b-instruct", temperature=0.0, **kwargs):
        super().__init__(**kwargs)
        self.custom_endpoint = custom_endpoint
        self.port = port
        self.model_name = model_name
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return 'llama'
        
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        response = self._call_custom_endpoint(messages)
        return self._create_chat_result(response)
    
    def _call_custom_endpoint(self, messages, **kwargs):
        import openai
        import json

        openai.api_key = "xyz"
        openai.base_url = "http://" + self.custom_endpoint + ":" + self.port + "/v1/"

        obj = json.loads(dumps(messages))
        
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": obj[0]["kwargs"]["content"]}], 
            temperature=self.temperature,
        )
        return response
    
    def _create_chat_result(self, response):
        from langchain_core.messages import ChatMessage
        from langchain_core.outputs import ChatResult, ChatGeneration
        
        message = ChatMessage(content=response.choices[0].message.content, role="assistant")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

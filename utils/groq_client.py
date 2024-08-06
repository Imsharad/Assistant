import os
from typing import List, Dict
from groq import Groq

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = os.environ["GROQ_MODEL_NAME"]

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
        )
        return chat_completion.choices[0].message.content

    def stream_response(self, messages: List[Dict[str, str]]):
        chat_completion = self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=True,
        )
        for chunk in chat_completion:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
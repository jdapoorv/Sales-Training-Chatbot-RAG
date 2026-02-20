import os
from typing import List, Optional

from openai import OpenAI
from google import genai

from src.interfaces import LLMProvider

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        self.model = model

    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434/v1"):
        self.client = OpenAI(api_key="ollama", base_url=base_url)
        self.model = model

    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate_content(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        full_prompt = f"{system_prompt}\n\nUSER INPUT:\n{user_prompt}"
        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config={'temperature': temperature}
        )
        return response.text.strip()

class ProviderFactory:
    @staticmethod
    def get_provider() -> LLMProvider:
        provider_type = os.getenv("LLM_PROVIDER", "openai").lower()
        
        if provider_type == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key: raise ValueError("Missing OPENAI_API_KEY")
            return OpenAIProvider(api_key=key, model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        
        elif provider_type == "groq":
            key = os.getenv("GROQ_API_KEY")
            if not key: raise ValueError("Missing GROQ_API_KEY")
            return GroqProvider(api_key=key, model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
        
        elif provider_type == "ollama":
            return OllamaProvider(
                model=os.getenv("OLLAMA_MODEL", "llama3"),
                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            )
        
        elif provider_type == "gemini":
            key = os.getenv("GEMINI_API_KEY")
            if not key: raise ValueError("Missing GEMINI_API_KEY")
            return GeminiProvider(api_key=key, model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest"))
        
        else:
            raise ValueError(f"Unsupported LLM_PROVIDER: {provider_type}")


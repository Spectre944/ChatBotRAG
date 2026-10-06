from .base import BaseLLM, build_prompt
from .claude_llm import ClaudeLLM
from .ollama_llm import OllamaLLM
from .openai_llm import OpenAILLM

__all__ = ["BaseLLM", "ClaudeLLM", "OpenAILLM", "OllamaLLM", "create_llm", "build_prompt"]


def create_llm(provider: str, model: str, **kwargs) -> BaseLLM:
    """
    Фабрика — создаёт нужный backend по строке из config.LLM_PROVIDER,
    не заставляя main.py/app.py знать о существовании конкретных классов.
    """
    providers = {
        "claude": ClaudeLLM,
        "openai": OpenAILLM,
        "ollama": OllamaLLM,
    }
    if provider not in providers:
        raise ValueError(f"Неизвестный провайдер '{provider}'. Доступны: {list(providers)}")
    return providers[provider](model=model, **kwargs)

import structlog
import pybreaker

from dataclasses import dataclass
from typing import Any, Callable, Optional

from tenacity import Retrying, stop_after_attempt, wait_exponential
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_cerebras import ChatCerebras
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq

from src.configurations.config import settings


logger = structlog.get_logger("llm")


@dataclass(frozen=True)
class LLMProvider:
    name: str
    model: str
    api_key: Optional[str]
    client: Callable[..., BaseChatModel]
    key_arg: str


class LLMManager:
    """
    Manages LLM provider selection, retries, circuit breakers, and fallback.

    Default provider order:
        Groq -> Cohere -> Cerebras

    Fallback mode:
        Cohere -> Cerebras
    """

    _breakers = {
        "groq": pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60),
        "cohere": pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60),
        "cerebras": pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60),
    }

    def __init__(self) -> None:
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        self.timeout = 8.0

        self.active_model: Optional[str] = None
        self.force_fallback = False

        self.providers = {
            "groq": LLMProvider(
                name="groq",
                model=settings.GROQ_MODEL,
                api_key=settings.GROQ_API_KEY,
                client=ChatGroq,
                key_arg="groq_api_key",
            ),
            "cohere": LLMProvider(
                name="cohere",
                model=settings.COHERE_MODEL,
                api_key=settings.COHERE_API_KEY,
                client=ChatCohere,
                key_arg="cohere_api_key",
            ),
            "cerebras": LLMProvider(
                name="cerebras",
                model=settings.CEREBRAS_MODEL,
                api_key=settings.CEREBRAS_API_KEY,
                client=ChatCerebras,
                key_arg="cerebras_api_key",
            ),
        }

    def get_llm(self, force_fallback: bool = False) -> "LLMManager":
        """
        Return this manager as the invokable LLM interface.

        Args:
            force_fallback: If True, skip Groq and start from fallback providers.
        """
        self.force_fallback = force_fallback
        return self

    def invoke(self, prompt: str, force_fallback: Optional[bool] = None) -> Any:
        """
        Invoke the LLM with retry, circuit breaker, and fallback support.

        Args:
            prompt: User prompt or LangChain-compatible input.
            force_fallback: If True, skip Groq and start from Cohere.

        Returns:
            LLM response.
        """
        use_fallback = (
            self.force_fallback
            if force_fallback is None
            else force_fallback
        )

        provider_order = self._provider_order(force_fallback=use_fallback)

        if not provider_order:
            raise RuntimeError("No LLM provider is configured.")

        retryer = self._retryer()
        last_error: Optional[Exception] = None

        for provider in provider_order:
            self.active_model = provider
            breaker = self._breakers[provider]

            try:
                return breaker.call(
                    retryer,
                    self._invoke_provider,
                    provider,
                    prompt,
                )

            except pybreaker.CircuitBreakerError as exc:
                logger.warning(
                    "LLM circuit breaker open. Provider skipped.",
                    provider=provider,
                    error=str(exc),
                )
                last_error = exc

            except Exception as exc:
                logger.error(
                    "LLM provider failed after retries.",
                    provider=provider,
                    error=str(exc),
                )
                last_error = exc

        raise RuntimeError(
            f"All configured LLM providers failed. Last error: {last_error}"
        )

    def get_active_model(self) -> Optional[str]:
        """
        Return the last attempted or currently active provider.
        """
        return self.active_model

    def _provider_order(self, force_fallback: bool = False) -> list[str]:
        order = (
            ["cohere", "cerebras"]
            if force_fallback
            else ["groq", "cohere", "cerebras"]
        )

        return [
            provider
            for provider in order
            if self.providers[provider].api_key
        ]

    def _invoke_provider(self, provider: str, prompt: str) -> Any:
        llm = self._build_provider(provider)
        return llm.invoke(prompt)

    def _build_provider(self, provider: str) -> BaseChatModel:
        config = self.providers[provider]

        if not config.api_key:
            raise RuntimeError(f"{provider} API key is not configured.")

        logger.info(
            "Initializing LLM provider",
            provider=config.name,
            model=config.model,
        )

        kwargs = {
            "model": config.model,
            config.key_arg: config.api_key,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        return config.client(**kwargs)

    def _retryer(self) -> Retrying:
        return Retrying(
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
            stop=stop_after_attempt(3),
            reraise=True,
            before_sleep=self._log_retry,
        )

    def _log_retry(self, retry_state: Any) -> None:
        logger.warning(
            "LLM retry scheduled",
            provider=self.active_model,
            attempt=retry_state.attempt_number,
            delay=retry_state.next_action.sleep,
            error=str(retry_state.outcome.exception()),
        )

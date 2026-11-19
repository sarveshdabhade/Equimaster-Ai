"""Optional LLM provider integrations."""

from .provider import (
    LLMDisabledError,
    LLMProvider,
    LLMProviderError,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    create_llm_provider,
)
from .evidence import synthesize_retrieved_evidence
from .evaluation import evaluate_llm_synthesis_dataset
from src.fusion_dataset import build_fusion_dataset, llm_feature_row_from_sentiment, validate_fusion_dataset, validate_fusion_record

__all__ = [
    "LLMDisabledError",
    "LLMProvider",
    "LLMProviderError",
    "OpenAICompatibleConfig",
    "OpenAICompatibleProvider",
    "create_llm_provider",
    "synthesize_retrieved_evidence",
    "evaluate_llm_synthesis_dataset",
    "build_fusion_dataset",
    "llm_feature_row_from_sentiment",
    "validate_fusion_dataset",
    "validate_fusion_record",
]

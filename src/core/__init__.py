"""
Модуль core содержит основную логику приложения.
"""

from .language_detector import LanguageDetector, create_language_detector
from .summarizer import SummaryResult, TextSummarizer, create_summarizer
from .text_processor import TextProcessor, create_text_processor

__all__ = [
    "TextSummarizer",
    "SummaryResult",
    "create_summarizer",
    "LanguageDetector",
    "create_language_detector",
    "TextProcessor",
    "create_text_processor",
]

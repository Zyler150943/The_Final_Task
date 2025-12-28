"""
Модуль core содержит основную логику приложения.
"""

from .summarizer import TextSummarizer, SummaryResult, create_summarizer
from .language_detector import LanguageDetector, create_language_detector
from .text_processor import TextProcessor, create_text_processor

__all__ = [
    'TextSummarizer',
    'SummaryResult',
    'create_summarizer',
    'LanguageDetector',
    'create_language_detector',
    'TextProcessor',
    'create_text_processor',
]
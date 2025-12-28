"""
Модуль src содержит основную функциональность приложения.
"""

__version__ = "1.0.0"
__author__ = "Zyler150943"
__repository__ = "https://github.com/Zyler150943/The_Final_Task"

# Реэкспорт основных компонентов для удобного импорта
from .core import (
    LanguageDetector,
    SummaryResult,
    TextProcessor,
    TextSummarizer,
    create_language_detector,
    create_summarizer,
    create_text_processor,
)

from .models.abstractive import (
    AbstractiveModel,
    BatchSummarizer,
    EnglishSummarizer,
    GermanSummarizer,
    ModelFactory,
    RussianSummarizer,
)

# Для обратной совместимости
Summarizer = TextSummarizer

__all__ = [
    # Основные компоненты
    "TextSummarizer",
    "Summarizer",
    "SummaryResult",
    "create_summarizer",
    "LanguageDetector",
    "create_language_detector",
    "TextProcessor",
    "create_text_processor",
    # Модели
    "AbstractiveModel",
    "EnglishSummarizer",
    "RussianSummarizer",
    "GermanSummarizer",
    "ModelFactory",
    "BatchSummarizer",
]
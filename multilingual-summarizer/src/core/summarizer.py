# -*- coding: utf-8 -*-
"""
Основной модуль для резюмирования текста на нескольких языках.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from nltk.tokenize import sent_tokenize
import nltk

# Локальные импорты
from .language_detector import LanguageDetector
from .text_processor import TextProcessor
from ..models.abstractive import ModelFactory, AbstractiveModel

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Класс для хранения результатов резюмирования."""
    
    original_text: str
    summary: str
    language: str
    compression: float
    key_points: List[str]
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time: Optional[float] = None


class TextSummarizer:
    """Класс для резюмирования текста на нескольких языках с поддержкой абстрактивных моделей."""
    
    # Конфигурация для разных языков
    LANGUAGE_CONFIGS = {
        'en': {
            'max_input_length': 1024,
            'max_output_length': 512,
            'min_output_length': 30,
        },
        'ru': {
            'max_input_length': 512,
            'max_output_length': 200,
            'min_output_length': 20,
        },
        'de': {
            'max_input_length': 512,
            'max_output_length': 200,
            'min_output_length': 20,
        }
    }
    
    # Поддержка языков и их коды
    LANGUAGE_MAP = {
        'english': 'en',
        'russian': 'ru',
        'german': 'de',
        'en': 'en',
        'ru': 'ru',
        'de': 'de'
    }
    
    # Уровни сжатия
    COMPRESSION_LEVELS = [20, 30, 50]
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_advanced_detector: bool = False,
        cache_models: bool = True
    ):
        """
        Инициализация суммаризатора.
        
        Args:
            device: Устройство для вычислений ('cuda', 'cpu', 'auto')
            use_advanced_detector: Использовать продвинутый детектор языка (FastText)
            cache_models: Кэшировать загруженные модели
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_models = cache_models
        
        # Инициализация компонентов
        self.language_detector = LanguageDetector(use_fasttext=use_advanced_detector)
        self.text_processor = TextProcessor(language='en')  # По умолчанию английский
        
        # Кэш моделей
        self.models: Dict[str, AbstractiveModel] = {}
        
        # Скачиваем необходимые ресурсы nltk
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            logger.warning(f"Не удалось загрузить ресурсы NLTK: {e}")
        
        logger.info(f"Инициализирован TextSummarizer на устройстве: {self.device}")
    
    def _normalize_language_code(self, language_input: str) -> Optional[str]:
        """
        Нормализация кода языка.
        
        Args:
            language_input: Входное значение (код или название языка)
        
        Returns:
            Нормализованный код языка или None
        """
        # Приводим к нижнему регистру
        language_input = language_input.lower()
        
        # Если это код, возвращаем его
        if language_input in ['en', 'ru', 'de']:
            return language_input
        
        # Если это название, преобразуем в код
        if language_input in ['english', 'russian', 'german']:
            return self.LANGUAGE_MAP[language_input]
        
        # Если это 'auto', возвращаем как есть
        if language_input == 'auto':
            return 'auto'
        
        return None
    
    def detect_language(self, text: str, use_advanced: bool = True) -> str:
        """
        Автоматическое определение языка текста.
        
        Args:
            text: Входной текст
            use_advanced: Использовать продвинутый детектор
        
        Returns:
            Код языка ('en', 'ru', 'de') или 'unknown'
        """
        # Используем наш детектор
        result = self.language_detector.detect_language(
            text,
            prefer_fasttext=use_advanced
        )
        
        if result['is_supported']:
            return result['language_code']
        return 'unknown'
    
    def _get_or_load_model(self, language_code: str) -> AbstractiveModel:
        """
        Получение или загрузка модели для языка.
        
        Args:
            language_code: Код языка
        
        Returns:
            Загруженная модель
        """
        # Проверяем, поддерживается ли язык
        if language_code not in self.LANGUAGE_CONFIGS:
            raise ValueError(f"Неподдерживаемый язык: {language_code}")
        
        # Проверяем кэш
        if self.cache_models and language_code in self.models:
            return self.models[language_code]
        
        # Загружаем модель
        logger.info(f"Загрузка модели для языка {language_code}...")
        
        try:
            model = ModelFactory.create_model(
                language=language_code,
                device=self.device
            )
            
            # Загружаем модель (ленивая загрузка)
            model.load()
            
            # Кэшируем если нужно
            if self.cache_models:
                self.models[language_code] = model
            
            logger.info(f"Модель для языка {language_code} успешно загружена")
            return model
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели для языка {language_code}: {e}")
            raise
    
    def _calculate_summary_lengths(
        self,
        text: str,
        language: str,
        compression: float
    ) -> Dict[str, int]:
        """
        Расчет параметров длин для резюмирования.
        
        Args:
            text: Входной текст
            language: Код языка
            compression: Уровень сжатия в процентах
        
        Returns:
            Словарь с параметрами длин
        """
        if language not in self.LANGUAGE_CONFIGS:
            raise ValueError(f"Неподдерживаемый язык: {language}")
        
        config = self.LANGUAGE_CONFIGS[language]
        
        # Приблизительная оценка длины текста в токенах
        words = text.split()
        approx_tokens = len(words) + len(text) // 10
        
        # Расчет желаемой длины резюме на основе уровня сжатия
        target_length = int(approx_tokens * compression / 100)
        
        # Ограничиваем максимальной и минимальной длиной
        max_output = config['max_output_length']
        min_output = config['min_output_length']
        target_length = max(min_output, min(target_length, max_output))
        
        return {
            'max_length': target_length,
            'min_length': int(target_length * 0.5),
            'max_input_length': config['max_input_length']
        }
    
    def _preprocess_for_summarization(
        self,
        text: str,
        language: str,
        max_input_length: int
    ) -> str:
        """
        Предобработка текста для резюмирования.
        
        Args:
            text: Исходный текст
            language: Код языка
            max_input_length: Максимальная длина входного текста
        
        Returns:
            Предобработанный текст
        """
        # Устанавливаем язык для процессора
        self.text_processor.language = language
        
        # Очищаем и подготавливаем текст
        cleaned_text = self.text_processor.clean_text(text)
        
        # Если текст очень длинный, разбиваем на части
        if len(cleaned_text.split()) > max_input_length:
            logger.warning(f"Текст слишком длинный, применяется обрезка")
            
            # Разбиваем на части
            parts = self.text_processor.split_long_text(
                cleaned_text,
                max_tokens=max_input_length
            )
            
            # Берем первую часть (самую важную)
            cleaned_text = parts[0] if parts else cleaned_text
        
        return cleaned_text
    
    def _extract_key_points(
        self,
        text: str,
        language: str,
        num_points: int = 5
    ) -> List[str]:
        """
        Извлечение ключевых моментов из текста.
        
        Args:
            text: Исходный текст
            language: Код языка
            num_points: Количество ключевых моментов
        
        Returns:
            Список ключевых моментов
        """
        try:
            # Устанавливаем язык для процессора
            self.text_processor.language = language
            
            # Извлекаем ключевые фразы
            key_phrases = self.text_processor.extract_key_phrases(
                text,
                top_n=num_points
            )
            
            # Форматируем результат
            key_points = []
            for i, (phrase, weight) in enumerate(key_phrases, 1):
                key_points.append(f"{i}. {phrase} (важность: {weight:.2f})")
            
            return key_points
            
        except Exception as e:
            logger.error(f"Ошибка извлечения ключевых моментов: {e}")
            
            # Fallback: простой метод
            try:
                sentences = sent_tokenize(text)
                return [f"{i+1}. {sentences[i]}" for i in range(min(num_points, len(sentences)))]
            except Exception:
                return ["Не удалось извлечь ключевые моменты"]
    
    def summarize(
        self,
        text: str,
        language: str = 'auto',
        compression: float = 30,
        abstractive: bool = True,
        extract_key_points: bool = True
    ) -> SummaryResult:
        """
        Основной метод для резюмирования текста.
        
        Args:
            text: Входной текст для резюмирования
            language: Язык текста ('en', 'ru', 'de', 'auto')
            compression: Уровень сжатия в процентах (20, 30, 50)
            abstractive: Использовать абстрактивное резюмирование (True) или экстрактивное (False)
            extract_key_points: Извлекать ключевые моменты
        
        Returns:
            SummaryResult с результатами резюмирования
        
        Raises:
            ValueError: При неподдерживаемом языке или некорректных параметрах
        """
        import time
        start_time = time.time()
        
        # Валидация входных параметров
        if not text or len(text.strip()) < 10:
            raise ValueError("Текст должен содержать минимум 10 символов")
        
        if compression not in self.COMPRESSION_LEVELS:
            raise ValueError(
                f"Уровень сжатия должен быть один из: {self.COMPRESSION_LEVELS}%"
            )
        
        # Нормализация языка
        normalized_language = self._normalize_language_code(language)
        if normalized_language is None:
            raise ValueError(f"Неподдерживаемый язык: {language}")
        
        # Определение языка, если указано 'auto'
        if normalized_language == 'auto':
            language_code = self.detect_language(text)
            if language_code == 'unknown':
                raise ValueError("Не удалось определить язык текста")
        else:
            language_code = normalized_language
        
        # Проверка поддержки языка
        if language_code not in self.LANGUAGE_CONFIGS:
            raise ValueError(f"Резюмирование на языке '{language_code}' не поддерживается")
        
        # Расчет параметров длин
        lengths = self._calculate_summary_lengths(text, language_code, compression)
        
        # Предобработка текста
        preprocessed_text = self._preprocess_for_summarization(
            text,
            language_code,
            lengths['max_input_length']
        )
        
        # Резюмирование
        if abstractive:
            # Абстрактивное резюмирование
            model = self._get_or_load_model(language_code)
            summary = model.summarize(
                preprocessed_text,
                max_length=lengths['max_length'],
                min_length=lengths['min_length']
            )
        else:
            # Экстрактивное резюмирование (простая реализация)
            summary = self._extractive_summarize(
                preprocessed_text,
                language_code,
                compression
            )
        
        # Извлечение ключевых моментов
        key_points = []
        if extract_key_points:
            key_points = self._extract_key_points(summary, language_code)
        
        # Расчет метрик
        original_length = len(text)
        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 0
        processing_time = time.time() - start_time
        
        # Создание результата
        result = SummaryResult(
            original_text=text,
            summary=summary,
            language=language_code,
            compression=compression,
            key_points=key_points,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            processing_time=processing_time
        )
        
        logger.info(
            f"Резюмирование завершено: {original_length} -> {summary_length} "
            f"символов (сжатие: {compression_ratio:.1%}, время: {processing_time:.2f}с)"
        )
        
        return result
    
    def _extractive_summarize(
        self,
        text: str,
        language: str,
        compression: float
    ) -> str:
        """
        Простое экстрактивное резюмирование.
        
        Args:
            text: Входной текст
            language: Код языка
            compression: Уровень сжатия в процентах
        
        Returns:
            Резюмированный текст
        """
        try:
            # Устанавливаем язык для процессора
            self.text_processor.language = language
            
            # Разбиваем на предложения
            sentences = self.text_processor.split_into_sentences(text)
            
            if not sentences:
                return text
            
            # Выбираем первые N предложений на основе уровня сжатия
            num_sentences = max(1, int(len(sentences) * compression / 100))
            selected_sentences = sentences[:num_sentences]
            
            # Объединяем выбранные предложения
            summary = ' '.join(selected_sentences)
            
            logger.info(f"Экстрактивное резюмирование: выбрано {num_sentences}/{len(sentences)} предложений")
            
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка экстрактивного резюмирования: {e}")
            # Fallback: возвращаем начало текста
            words = text.split()
            num_words = max(10, int(len(words) * compression / 100))
            return ' '.join(words[:num_words])
    
    def summarize_batch(
        self,
        texts: List[str],
        language: str = 'auto',
        compression: float = 30,
        batch_size: int = 4,
        show_progress: bool = True
    ) -> List[SummaryResult]:
        """
        Пакетное резюмирование текстов.
        
        Args:
            texts: Список текстов для резюмирования
            language: Язык текстов
            compression: Уровень сжатия
            batch_size: Размер пакета
            show_progress: Показывать прогресс
        
        Returns:
            Список результатов резюмирования
        """
        from tqdm import tqdm
        
        results = []
        
        # Определяем язык для всех текстов
        language_code = None
        if language != 'auto':
            normalized_language = self._normalize_language_code(language)
            if normalized_language and normalized_language != 'auto':
                language_code = normalized_language
        
        # Итерация по текстам
        iterator = range(len(texts))
        if show_progress:
            iterator = tqdm(iterator, desc="Пакетное резюмирование")
        
        for i in iterator:
            try:
                # Определяем язык для текущего текста если нужно
                current_language = language_code
                if not current_language:
                    detected_lang = self.detect_language(texts[i])
                    if detected_lang != 'unknown':
                        current_language = detected_lang
                    else:
                        # Используем язык по умолчанию
                        current_language = 'en'
                
                # Резюмируем текст
                result = self.summarize(
                    text=texts[i],
                    language=current_language,
                    compression=compression,
                    abstractive=True,
                    extract_key_points=True
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке текста {i+1}: {e}")
                # Добавляем пустой результат в случае ошибки
                results.append(None)
        
        return results
    
    def clear_model_cache(self):
        """Очистка кэша моделей."""
        self.models.clear()
        logger.info("Кэш моделей очищен")
    
    def get_supported_languages(self) -> List[str]:
        """Получение списка поддерживаемых языков."""
        return list(self.LANGUAGE_CONFIGS.keys())
    
    def get_compression_levels(self) -> List[float]:
        """Получение доступных уровней сжатия."""
        return self.COMPRESSION_LEVELS


# Фабричная функция для удобства
def create_summarizer(
    device: Optional[str] = None,
    use_advanced_detector: bool = False,
    cache_models: bool = True
) -> TextSummarizer:
    """
    Создание экземпляра суммаризатора.
    
    Args:
        device: Устройство для вычислений
        use_advanced_detector: Использовать продвинутый детектор языка
        cache_models: Кэшировать загруженные модели
    
    Returns:
        Экземпляр TextSummarizer
    """
    return TextSummarizer(
        device=device,
        use_advanced_detector=use_advanced_detector,
        cache_models=cache_models
    )
# -*- coding: utf-8 -*-
"""
Модуль для предобработки текста перед резюмированием.
"""

import re
import string
import logging
from typing import List, Dict, Optional, Tuple
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextProcessor:
    """
    Класс для предобработки текста на нескольких языках.
    Поддерживает английский, русский и немецкий языки.
    """

    # Стоп-слова для разных языков
    STOPWORDS = {
        'en': set(stopwords.words('english')),
        'ru': set(stopwords.words('russian')),
        'de': set(stopwords.words('german')),
    }

    # Регулярные выражения для очистки текста
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
    PHONE_PATTERN = re.compile(r'\+?\d[\d\s\-\(\)]{7,}\d')
    SPECIAL_CHARS_PATTERN = re.compile(r'[^\w\s.,!?;:()\-]')
    MULTIPLE_SPACES_PATTERN = re.compile(r'\s+')
    MULTIPLE_NEWLINES_PATTERN = re.compile(r'\n{3,}')

    # Знаки препинания для разных языков
    PUNCTUATION = {
        'en': string.punctuation,
        'ru': string.punctuation + '«»—',
        'de': string.punctuation + '«»—',
    }

    def __init__(self, language: str = 'en'):
        """
        Инициализация процессора текста.

        Args:
            language: Код языка ('en', 'ru', 'de')
        """
        self.language = language
        
        # Загрузка необходимых ресурсов NLTK
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Не удалось загрузить ресурсы NLTK: {e}")
        
        logger.info(f"Инициализирован TextProcessor для языка: {language}")

    def clean_text(self, text: str, remove_stopwords: bool = False) -> str:
        """
        Очистка текста от лишних символов и нормализация.

        Args:
            text: Исходный текст
            remove_stopwords: Удалять стоп-слова

        Returns:
            Очищенный текст
        """
        if not text:
            return ""

        # Сохраняем оригинал для логов
        original_length = len(text)

        # Удаляем URLs
        text = self.URL_PATTERN.sub('', text)

        # Удаляем email адреса
        text = self.EMAIL_PATTERN.sub('', text)

        # Удаляем номера телефонов
        text = self.PHONE_PATTERN.sub('', text)

        # Заменяем множественные переносы строк на двойные
        text = self.MULTIPLE_NEWLINES_PATTERN.sub('\n\n', text)

        # Удаляем специальные символы, но сохраняем основные знаки препинания
        text = self.SPECIAL_CHARS_PATTERN.sub(' ', text)

        # Удаляем лишние пробелы
        text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text)

        # Удаляем стоп-слова если требуется
        if remove_stopwords:
            text = self._remove_stopwords(text)

        # Удаляем лишние пробелы в начале и конце
        text = text.strip()

        # Логируем результат очистки
        cleaned_length = len(text)
        logger.debug(f"Очистка текста: {original_length} -> {cleaned_length} символов")

        return text

    def _remove_stopwords(self, text: str) -> str:
        """
        Удаление стоп-слов из текста.

        Args:
            text: Исходный текст

        Returns:
            Текст без стоп-слов
        """
        if self.language not in self.STOPWORDS:
            return text

        stop_words = self.STOPWORDS[self.language]
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_words)

    def split_into_sentences(self, text: str) -> List[str]:
        """
        Разбиение текста на предложения.

        Args:
            text: Исходный текст

        Returns:
            Список предложений
        """
        try:
            sentences = sent_tokenize(text, language=self.language)
            return [s.strip() for s in sentences if s.strip()]
        except Exception as e:
            logger.error(f"Ошибка при разбиении на предложения: {e}")
            # Fallback: разбиение по знакам препинания
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]

    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Разбиение текста на абзацы.

        Args:
            text: Исходный текст

        Returns:
            Список абзацев
        """
        paragraphs = text.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]

    def tokenize_words(self, text: str) -> List[str]:
        """
        Токенизация текста на слова.

        Args:
            text: Исходный текст

        Returns:
            Список слов
        """
        try:
            words = word_tokenize(text, language=self.language)
            return [word for word in words if word.strip()]
        except Exception as e:
            logger.error(f"Ошибка при токенизации слов: {e}")
            # Fallback: простая токенизация
            return text.split()

    def calculate_text_statistics(self, text: str) -> Dict[str, any]:
        """
        Расчет статистики текста.

        Args:
            text: Исходный текст

        Returns:
            Словарь со статистикой
        """
        # Разбиваем на предложения и слова
        sentences = self.split_into_sentences(text)
        words = self.tokenize_words(text)

        # Уникальные слова
        unique_words = set(word.lower() for word in words)

        # Длины
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = np.mean([len(word) for word in words]) if words else 0

        # Частота слов
        word_freq = Counter(words)

        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'unique_words': len(unique_words),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'avg_word_length': round(avg_word_length, 2),
            'lexical_diversity': len(unique_words) / len(words) if words else 0,
            'most_common_words': word_freq.most_common(10),
        }

    def normalize_text(self, text: str) -> str:
        """
        Нормализация текста: приведение к нижнему регистру,
        удаление лишних пробелов и знаков препинания.

        Args:
            text: Исходный текст

        Returns:
            Нормализованный текст
        """
        # Приводим к нижнему регистру
        text = text.lower()

        # Удаляем знаки препинания
        if self.language in self.PUNCTUATION:
            punctuation = self.PUNCTUATION[self.language]
            text = text.translate(str.maketrans('', '', punctuation))

        # Удаляем цифры
        text = re.sub(r'\d+', '', text)

        # Удаляем лишние пробелы
        text = self.MULTIPLE_SPACES_PATTERN.sub(' ', text)

        return text.strip()

    def extract_key_phrases(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Извлечение ключевых фраз из текста.

        Args:
            text: Исходный текст
            top_n: Количество извлекаемых фраз

        Returns:
            Список кортежей (фраза, вес)
        """
        # Очищаем и нормализуем текст
        cleaned_text = self.clean_text(text, remove_stopwords=True)
        normalized_text = self.normalize_text(cleaned_text)

        # Разбиваем на слова
        words = self.tokenize_words(normalized_text)

        # Создаем биграммы
        bigrams = list(nltk.bigrams(words))

        # Подсчитываем частоту
        bigram_freq = nltk.FreqDist(bigrams)

        # Вычисляем вес фраз (простота для примера)
        key_phrases = []
        for bigram, freq in bigram_freq.most_common(top_n):
            phrase = ' '.join(bigram)
            # Простой вес на основе частоты
            weight = freq / len(bigrams) if bigrams else 0
            key_phrases.append((phrase, weight))

        return key_phrases

    def split_long_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Разделение длинного текста на части для обработки.

        Args:
            text: Исходный текст
            max_tokens: Максимальное количество токенов в части

        Returns:
            Список частей текста
        """
        # Разбиваем на предложения
        sentences = self.split_into_sentences(text)

        if not sentences:
            return [text]

        # Объединяем предложения в части
        parts = []
        current_part = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenize_words(sentence))

            # Если текущая часть пустая или предложение помещается
            if not current_part or current_length + sentence_tokens <= max_tokens:
                current_part.append(sentence)
                current_length += sentence_tokens
            else:
                # Сохраняем текущую часть и начинаем новую
                parts.append(' '.join(current_part))
                current_part = [sentence]
                current_length = sentence_tokens

        # Добавляем последнюю часть
        if current_part:
            parts.append(' '.join(current_part))

        logger.info(f"Текст разделен на {len(parts)} частей")
        return parts

    def prepare_for_summarization(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Подготовка текста для резюмирования.

        Args:
            text: Исходный текст
            max_length: Максимальная длина текста

        Returns:
            Подготовленный текст
        """
        # Очищаем текст
        cleaned_text = self.clean_text(text)

        # Обрезаем если нужно
        if max_length and len(cleaned_text) > max_length:
            logger.warning(f"Текст обрезан до {max_length} символов")
            cleaned_text = cleaned_text[:max_length]

        # Убедимся, что текст заканчивается полным предложением
        if cleaned_text and cleaned_text[-1] not in '.!?':
            # Находим последнее полное предложение
            sentences = self.split_into_sentences(cleaned_text)
            if sentences:
                cleaned_text = ' '.join(sentences)
            else:
                cleaned_text = cleaned_text.rstrip() + '.'

        return cleaned_text

    def detect_language_from_text(self, text: str) -> str:
        """
        Простое определение языка по символам.

        Args:
            text: Исходный текст

        Returns:
            Предполагаемый код языка
        """
        # Простая эвристика на основе символов
        cyrillic_count = sum(1 for char in text if '\u0400' <= char <= '\u04FF')
        german_chars = sum(1 for char in text if char in 'äöüßÄÖÜ')

        # Если больше 30% кириллицы - вероятно русский
        if len(text) > 0 and cyrillic_count / len(text) > 0.3:
            return 'ru'
        # Если есть немецкие специфические символы
        elif german_chars > 2:
            return 'de'
        # По умолчанию английский
        else:
            return 'en'


# Фабричная функция для удобства
def create_text_processor(language: str = 'en') -> TextProcessor:
    """
    Создание экземпляра TextProcessor.

    Args:
        language: Код языка

    Returns:
        Экземпляр TextProcessor
    """
    return TextProcessor(language=language)


# Пример использования
if __name__ == "__main__":
    # Пример текста на английском
    sample_text = """
    Machine learning is a subset of artificial intelligence that focuses on developing algorithms 
    that can learn from data and make predictions. The main types of machine learning include 
    supervised learning, unsupervised learning, and reinforcement learning.
    
    Supervised learning involves training a model on labeled data. The model learns from examples 
    that have both input data and the correct output. This is similar to how a student learns with 
    a teacher who provides answers and feedback.
    
    For more information, visit https://example.com or contact us at info@example.com.
    Call us at +1 (555) 123-4567 for details.
    """

    # Создание процессора
    processor = TextProcessor(language='en')

    # Очистка текста
    cleaned = processor.clean_text(sample_text)
    print("=== Очищенный текст ===")
    print(cleaned[:200] + "...")
    print()

    # Статистика
    stats = processor.calculate_text_statistics(cleaned)
    print("=== Статистика текста ===")
    for key, value in stats.items():
        if key != 'most_common_words':
            print(f"{key}: {value}")
    print()

    # Разбиение на предложения
    sentences = processor.split_into_sentences(cleaned)
    print(f"=== Предложения ({len(sentences)}) ===")
    for i, sentence in enumerate(sentences[:3], 1):
        print(f"{i}. {sentence[:50]}...")
    print()

    # Ключевые фразы
    key_phrases = processor.extract_key_phrases(cleaned, top_n=5)
    print("=== Ключевые фразы ===")
    for phrase, weight in key_phrases:
        print(f"- {phrase} (вес: {weight:.3f})")
    print()

    # Подготовка для резюмирования
    prepared = processor.prepare_for_summarization(cleaned, max_length=500)
    print("=== Подготовленный для резюмирования текст ===")
    print(prepared[:200] + "...")
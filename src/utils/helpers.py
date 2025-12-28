# -*- coding: utf-8 -*-
"""
Вспомогательные функции.
"""

import hashlib
import re
import time
import unicodedata
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def format_size(size_bytes: int) -> str:
    """
    Форматирование размера в байтах в читаемый вид.

    Args:
        size_bytes: Размер в байтах

    Returns:
        Отформатированная строка
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Расчет коэффициента сжатия.

    Args:
        original_size: Исходный размер
        compressed_size: Размер после сжатия

    Returns:
        Коэффициент сжатия (0-1)
    """
    if original_size == 0:
        return 0.0
    return compressed_size / original_size


def validate_text(text: str, min_length: int = 10, max_length: int = 100000) -> bool:
    """
    Валидация текста.

    Args:
        text: Текст для валидации
        min_length: Минимальная длина
        max_length: Максимальная длина

    Returns:
        True если текст валиден
    """
    if not text or not isinstance(text, str):
        return False

    text_length = len(text.strip())

    if text_length < min_length:
        return False

    if text_length > max_length:
        return False

    # Проверяем, что текст содержит осмысленные символы
    # (не только пробелы и знаки препинания)
    meaningful_chars = re.sub(r"[\s\.,!?;:()\-]", "", text)
    if len(meaningful_chars) < min_length // 2:
        return False

    return True


def normalize_text(text: str) -> str:
    """
    Нормализация текста (удаление лишних пробелов, нормализация Unicode).

    Args:
        text: Исходный текст

    Returns:
        Нормализованный текст
    """
    # Нормализация Unicode
    text = unicodedata.normalize("NFKC", text)

    # Замена различных типов пробелов на обычные пробелы
    text = re.sub(r"[\t\n\r\f\v]", " ", text)

    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text)

    # Удаление пробелов в начале и конце
    text = text.strip()

    return text


def generate_text_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Генерация хеша текста.

    Args:
        text: Исходный текст
        algorithm: Алгоритм хеширования

    Returns:
        Хеш текста
    """
    text_bytes = text.encode("utf-8")

    if algorithm == "md5":
        return hashlib.md5(text_bytes).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(text_bytes).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(text_bytes).hexdigest()
    else:
        raise ValueError(f"Неподдерживаемый алгоритм: {algorithm}")


def timeit_decorator(func: Callable) -> Callable:
    """
    Декоратор для измерения времени выполнения функции.

    Args:
        func: Функция для измерения

    Returns:
        Обернутая функция
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Функция {func.__name__} выполнена за {execution_time:.4f} секунд")

        return result

    return wrapper


class Timer:
    """
    Класс для измерения времени выполнения блока кода.
    """

    def __init__(self, name: str = ""):
        """
        Инициализация таймера.

        Args:
            name: Имя таймера
        """
        self.name = name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Начало измерения времени."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Конец измерения времени."""
        self.end_time = time.time()

        if self.name:
            print(f"{self.name}: {self.elapsed:.4f} секунд")
        else:
            print(f"Время выполнения: {self.elapsed:.4f} секунд")

    @property
    def elapsed(self) -> float:
        """
        Получение прошедшего времени.

        Returns:
            Время в секундах
        """
        if self.start_time is None:
            return 0.0

        if self.end_time is None:
            return time.time() - self.start_time

        return self.end_time - self.start_time


def split_text_by_length(text: str, max_length: int) -> List[str]:
    """
    Разделение текста на части по максимальной длине.

    Args:
        text: Исходный текст
        max_length: Максимальная длина части

    Returns:
        Список частей текста
    """
    if len(text) <= max_length:
        return [text]

    parts = []
    current_part = []
    current_length = 0

    # Разбиваем по предложениям если возможно
    sentences = re.split(r"(?<=[.!?])\s+", text)

    for sentence in sentences:
        sentence_length = len(sentence)

        if current_length + sentence_length <= max_length:
            current_part.append(sentence)
            current_length += sentence_length
        else:
            if current_part:
                parts.append(" ".join(current_part))

            # Если одно предложение длиннее max_length
            if sentence_length > max_length:
                # Разбиваем на слова
                words = sentence.split()
                current_part = []
                current_length = 0

                for word in words:
                    word_length = len(word) + 1  # +1 для пробела

                    if current_length + word_length <= max_length:
                        current_part.append(word)
                        current_length += word_length
                    else:
                        if current_part:
                            parts.append(" ".join(current_part))
                        current_part = [word]
                        current_length = word_length
            else:
                current_part = [sentence]
                current_length = sentence_length

    # Добавляем последнюю часть
    if current_part:
        parts.append(" ".join(current_part))

    return parts


def get_file_extension(file_path: str) -> str:
    """
    Получение расширения файла.

    Args:
        file_path: Путь к файлу

    Returns:
        Расширение файла (в нижнем регистре, без точки)
    """
    return Path(file_path).suffix.lower()[1:] if Path(file_path).suffix else ""


def safe_get(dictionary: Dict, *keys, default: Any = None) -> Any:
    """
    Безопасное получение значения из вложенного словаря.

    Args:
        dictionary: Исходный словарь
        *keys: Ключи для последовательного доступа
        default: Значение по умолчанию

    Returns:
        Значение или default
    """
    current = dictionary

    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default

    return current


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Форматирование числа в процентное представление.

    Args:
        value: Значение (0-1)
        decimals: Количество знаков после запятой

    Returns:
        Отформатированная строка
    """
    percentage = value * 100
    return f"{percentage:.{decimals}f}%"

# -*- coding: utf-8 -*-
"""
Модуль для работы с файлами.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chardet

# Настройка логирования
logger = logging.getLogger(__name__)


def load_text_file(file_path: str, encoding: Optional[str] = None) -> str:
    """
    Загрузка текстового файла с автоматическим определением кодировки.

    Args:
        file_path: Путь к файлу
        encoding: Кодировка файла (если None, определяется автоматически)

    Returns:
        Содержимое файла в виде строки

    Raises:
        FileNotFoundError: Если файл не найден
        IOError: Если произошла ошибка чтения
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    if not file_path.is_file():
        raise IOError(f"Указанный путь не является файлом: {file_path}")

    # Определяем кодировку если не указана
    if encoding is None:
        encoding = detect_file_encoding(file_path)

    try:
        with open(file_path, "r", encoding=encoding) as file:
            content = file.read()

        logger.info(f"Файл загружен: {file_path}, размер: {len(content)} символов")
        return content

    except UnicodeDecodeError as e:
        logger.error(f"Ошибка декодирования файла {file_path}: {e}")
        # Пробуем другие кодировки
        for alt_encoding in ["utf-8-sig", "latin-1", "cp1251", "cp1252"]:
            try:
                with open(file_path, "r", encoding=alt_encoding) as file:
                    content = file.read()
                logger.info(f"Файл загружен с кодировкой {alt_encoding}: {file_path}")
                return content
            except UnicodeDecodeError:
                continue

        raise IOError(f"Не удалось декодировать файл {file_path}")

    except Exception as e:
        logger.error(f"Ошибка чтения файла {file_path}: {e}")
        raise


def save_text_file(
    content: str, file_path: str, encoding: str = "utf-8", overwrite: bool = False
) -> bool:
    """
    Сохранение текста в файл.

    Args:
        content: Текст для сохранения
        file_path: Путь к файлу
        encoding: Кодировка файла
        overwrite: Перезаписывать существующий файл

    Returns:
        True если успешно, False в противном случае
    """
    file_path = Path(file_path)

    # Проверяем, существует ли файл
    if file_path.exists() and not overwrite:
        logger.warning(f"Файл уже существует: {file_path}")
        return False

    try:
        # Создаем директории если нужно
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding=encoding) as file:
            file.write(content)

        logger.info(f"Файл сохранен: {file_path}, размер: {len(content)} символов")
        return True

    except Exception as e:
        logger.error(f"Ошибка сохранения файла {file_path}: {e}")
        return False


def detect_file_encoding(file_path: str) -> str:
    """
    Определение кодировки файла.

    Args:
        file_path: Путь к файлу

    Returns:
        Предполагаемая кодировка
    """
    file_path = Path(file_path)

    try:
        # Читаем первые 10KB файла для определения кодировки
        with open(file_path, "rb") as file:
            raw_data = file.read(10240)

        if not raw_data:
            return "utf-8"

        result = chardet.detect(raw_data)
        encoding = result.get("encoding", "utf-8")
        confidence = result.get("confidence", 0)

        logger.debug(f"Определена кодировка: {encoding}, уверенность: {confidence:.2f}")

        # Если уверенность низкая, используем utf-8
        if confidence < 0.5:
            return "utf-8"

        # Нормализация названий кодировок
        encoding_map = {
            "ascii": "utf-8",
            "Windows-1251": "cp1251",
            "Windows-1252": "cp1252",
        }

        return encoding_map.get(encoding, encoding)

    except Exception as e:
        logger.error(f"Ошибка определения кодировки файла {file_path}: {e}")
        return "utf-8"


def load_config_file(file_path: str) -> Dict[str, Any]:
    """
    Загрузка конфигурационного файла (JSON или YAML).

    Args:
        file_path: Путь к конфигурационному файлу

    Returns:
        Словарь с конфигурацией
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"Конфигурационный файл не найден: {file_path}")
        return {}

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Определяем формат по расширению
        if file_path.suffix.lower() == ".json":
            return json.loads(content)
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            try:
                import yaml

                return yaml.safe_load(content)
            except ImportError:
                logger.error("Для загрузки YAML файлов установите pyyaml")
                return {}
        else:
            logger.error(
                f"Неподдерживаемый формат конфигурационного файла: {file_path.suffix}"
            )
            return {}

    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурационного файла {file_path}: {e}")
        return {}


def get_file_info(file_path: str) -> Dict[str, Any]:
    """
    Получение информации о файле.

    Args:
        file_path: Путь к файлу

    Returns:
        Словарь с информацией о файле
    """
    file_path = Path(file_path)

    if not file_path.exists():
        return {}

    try:
        stat = file_path.stat()
        encoding = detect_file_encoding(file_path)

        # Пытаемся определить тип файла
        file_type = "unknown"
        if file_path.suffix.lower() in [".txt", ".md", ".rst"]:
            file_type = "text"
        elif file_path.suffix.lower() in [".json", ".yaml", ".yml", ".xml"]:
            file_type = "structured"
        elif file_path.suffix.lower() in [".pdf", ".docx", ".odt"]:
            file_type = "document"

        return {
            "path": str(file_path),
            "name": file_path.name,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "encoding": encoding,
            "type": file_type,
            "extension": file_path.suffix.lower(),
        }

    except Exception as e:
        logger.error(f"Ошибка получения информации о файле {file_path}: {e}")
        return {}


def list_text_files(
    directory: str, extensions: Optional[List[str]] = None
) -> List[str]:
    """
    Поиск текстовых файлов в директории.

    Args:
        directory: Директория для поиска
        extensions: Список расширений файлов

    Returns:
        Список путей к текстовым файлам
    """
    if extensions is None:
        extensions = [".txt", ".md", ".rst", ".json", ".yaml", ".yml"]

    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        logger.error(f"Директория не найдена: {directory}")
        return []

    try:
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
            files.extend(directory.glob(f"**/*{ext}"))

        # Убираем дубликаты и сортируем
        files = sorted(set(files))

        logger.info(f"Найдено {len(files)} файлов в {directory}")
        return [str(f) for f in files]

    except Exception as e:
        logger.error(f"Ошибка поиска файлов в {directory}: {e}")
        return []


def batch_process_files(
    input_dir: str,
    output_dir: str,
    process_function,
    file_extensions: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    Пакетная обработка файлов.

    Args:
        input_dir: Входная директория
        output_dir: Выходная директория
        process_function: Функция для обработки файлов
        file_extensions: Расширения файлов для обработки
        overwrite: Перезаписывать существующие файлы

    Returns:
        Статистика обработки
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        logger.error(f"Входная директория не найдена: {input_dir}")
        return {"success": False, "error": "Input directory not found"}

    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)

    # Находим файлы
    files = list_text_files(str(input_dir), file_extensions)

    if not files:
        logger.warning(f"Файлы не найдены в {input_dir}")
        return {"processed": 0, "failed": 0, "skipped": 0}

    # Статистика
    stats = {
        "total": len(files),
        "processed": 0,
        "failed": 0,
        "skipped": 0,
        "results": [],
    }

    # Обрабатываем файлы
    for file_path in files:
        try:
            input_file = Path(file_path)
            # Создаем путь для выходного файла
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path.with_suffix(".txt")

            # Проверяем, нужно ли обрабатывать
            if output_file.exists() and not overwrite:
                logger.info(f"Файл уже существует, пропускаем: {output_file}")
                stats["skipped"] += 1
                continue

            # Загружаем файл
            content = load_text_file(input_file)

            # Обрабатываем
            result = process_function(content)

            # Сохраняем результат
            save_text_file(result, output_file, overwrite=overwrite)

            stats["processed"] += 1
            stats["results"].append(
                {"input": str(input_file), "output": str(output_file), "success": True}
            )

            logger.info(f"Обработан файл: {input_file} -> {output_file}")

        except Exception as e:
            logger.error(f"Ошибка обработки файла {file_path}: {e}")
            stats["failed"] += 1
            stats["results"].append(
                {"input": file_path, "error": str(e), "success": False}
            )

    logger.info(
        f"Обработка завершена: {stats['processed']}/{stats['total']} "
        f"(ошибок: {stats['failed']}, пропущено: {stats['skipped']})"
    )

    return stats

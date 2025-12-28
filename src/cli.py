#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Командный интерфейс для Multilingual Learning Material Summarizer.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Добавляем путь для импорта модулей
sys.path.append(str(Path(__file__).parent))

from core import create_summarizer
from utils.config import create_config_file, load_config
from utils.file_handler import (
    batch_process_files,
    list_text_files,
    load_text_file,
    save_text_file,
)
from utils.helpers import format_size, validate_text
from utils.logger import setup_logging


def setup_argparse() -> argparse.ArgumentParser:
    """
    Настройка парсера аргументов командной строки.

    Returns:
        Настроенный парсер аргументов
    """
    parser = argparse.ArgumentParser(
        description="Multilingual Learning Material Summarizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Резюмирование файла
  python cli.py --input data/lecture.txt --language en --compression 30
  
  # Резюмирование текста напрямую
  python cli.py --text "Ваш текст здесь..." --compression 20 --output result.txt
  
  # Пакетная обработка файлов
  python cli.py --batch data/lectures/ --output results/ --language auto
  
  # Генерация конфигурационного файла
  python cli.py --generate-config config.json
  
  # Создание примера учебного материала
  python cli.py --create-sample
        """,
    )

    # Основные параметры
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument("--input", "-i", type=str, help="Путь к входному файлу")
    input_group.add_argument(
        "--text", "-t", type=str, help="Текст для резюмирования (если не указан файл)"
    )
    input_group.add_argument(
        "--batch", "-b", type=str, help="Директория для пакетной обработки файлов"
    )

    # Параметры вывода
    parser.add_argument(
        "--output", "-o", type=str, help="Путь к выходному файлу или директории"
    )

    # Параметры резюмирования
    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="auto",
        choices=["auto", "en", "ru", "de", "english", "russian", "german"],
        help="Язык текста (по умолчанию: auto)",
    )

    parser.add_argument(
        "--compression",
        "-c",
        type=int,
        default=30,
        choices=[20, 30, 50],
        help="Уровень сжатия в процентах (по умолчанию: 30)",
    )

    parser.add_argument(
        "--abstractive",
        action="store_true",
        default=True,
        help="Использовать абстрактивное резюмирование (по умолчанию: True)",
    )

    parser.add_argument(
        "--no-abstractive",
        action="store_false",
        dest="abstractive",
        help="Использовать экстрактивное резюмирование",
    )

    parser.add_argument(
        "--key-points",
        action="store_true",
        default=True,
        help="Извлекать ключевые моменты (по умолчанию: True)",
    )

    parser.add_argument(
        "--no-key-points",
        action="store_false",
        dest="key_points",
        help="Не извлекать ключевые моменты",
    )

    # Параметры системы
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Путь к конфигурационному файлу",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Устройство для вычислений",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Уровень детализации логов",
    )

    parser.add_argument("--log-file", type=str, help="Файл для сохранения логов")

    # Утилитарные параметры
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Создать конфигурационный файл по умолчанию",
    )

    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Создать примеры учебных материалов",
    )

    parser.add_argument(
        "--list-languages", action="store_true", help="Показать поддерживаемые языки"
    )

    parser.add_argument(
        "--version", action="version", version="Multilingual Summarizer 1.0.0"
    )

    return parser


def process_single_file(
    input_path: str,
    output_path: Optional[str],
    language: str,
    compression: int,
    abstractive: bool,
    extract_key_points: bool,
    summarizer,
    logger,
) -> bool:
    """
    Обработка одиночного файла.

    Args:
        input_path: Путь к входному файлу
        output_path: Путь к выходному файлу
        language: Язык текста
        compression: Уровень сжатия
        abstractive: Использовать абстрактивное резюмирование
        extract_key_points: Извлекать ключевые моменты
        summarizer: Экземпляр суммаризатора
        logger: Логгер

    Returns:
        True если успешно, False в противном случае
    """
    try:
        # Загрузка файла
        text = load_text_file(input_path)

        # Валидация текста
        if not validate_text(text):
            logger.error(f"Текст в файле {input_path} слишком короткий или невалидный")
            return False

        # Резюмирование
        result = summarizer.summarize(
            text=text,
            language=language,
            compression=compression,
            abstractive=abstractive,
            extract_key_points=extract_key_points,
        )

        # Формирование результата
        output_content = format_result(result, input_path)

        # Сохранение или вывод
        if output_path:
            save_text_file(output_content, output_path, overwrite=True)
            logger.info(f"Результат сохранен в {output_path}")
            logger.info(f"Статистика: {len(text)} -> {len(result.summary)} символов")
        else:
            print(output_content)

        return True

    except Exception as e:
        logger.error(f"Ошибка обработки файла {input_path}: {e}")
        return False


def process_batch(
    input_dir: str,
    output_dir: str,
    language: str,
    compression: int,
    abstractive: bool,
    extract_key_points: bool,
    summarizer,
    logger,
) -> dict:
    """
    Пакетная обработка файлов в директории.

    Args:
        input_dir: Входная директория
        output_dir: Выходная директория
        language: Язык текста
        compression: Уровень сжатия
        abstractive: Использовать абстрактивное резюмирование
        extract_key_points: Извлекать ключевые моменты
        summarizer: Экземпляр суммаризатора
        logger: Логгер

    Returns:
        Статистика обработки
    """

    def process_function(text: str) -> str:
        """Функция для обработки одного текста."""
        result = summarizer.summarize(
            text=text,
            language=language,
            compression=compression,
            abstractive=abstractive,
            extract_key_points=extract_key_points,
        )
        return format_result(result, "batch_file")

    # Запуск пакетной обработки
    stats = batch_process_files(
        input_dir=input_dir,
        output_dir=output_dir,
        process_function=process_function,
        file_extensions=[".txt", ".md"],
        overwrite=True,
    )

    return stats


def format_result(result, source_name: str = "") -> str:
    """
    Форматирование результата резюмирования.

    Args:
        result: Результат резюмирования
        source_name: Имя исходного файла

    Returns:
        Отформатированная строка с результатом
    """
    lines = []

    # Заголовок
    if source_name:
        lines.append(f"=== Резюме: {source_name} ===")
    else:
        lines.append("=== Резюме ===")

    lines.append(f"Язык: {result.language}")
    lines.append(f"Уровень сжатия: {result.compression}%")
    lines.append(f"Исходный размер: {result.original_length} символов")
    lines.append(f"Размер резюме: {result.summary_length} символов")
    lines.append(f"Коэффициент сжатия: {result.compression_ratio:.1%}")

    if hasattr(result, "processing_time") and result.processing_time:
        lines.append(f"Время обработки: {result.processing_time:.2f}с")

    lines.append("")
    lines.append("=== Текст резюме ===")
    lines.append(result.summary)
    lines.append("")

    if result.key_points:
        lines.append("=== Ключевые моменты ===")
        for point in result.key_points:
            lines.append(f"  • {point}")

    lines.append("=" * 50)

    return "\n".join(lines)


def create_sample_materials(output_dir: str = "data") -> None:
    """
    Создание примеров учебных материалов.

    Args:
        output_dir: Директория для сохранения примеров
    """
    samples = {
        "sample_english.txt": """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from data and make predictions. The main types of machine learning include supervised learning, unsupervised learning, and reinforcement learning.

## Supervised Learning

Supervised learning involves training a model on labeled data. The model learns from examples that have both input data and the correct output. Common algorithms include linear regression, logistic regression, and support vector machines.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. The model tries to learn the structure of the data without explicit guidance. Common algorithms include k-means clustering and principal component analysis.

## Reinforcement Learning

Reinforcement learning uses rewards and punishments to train models. The agent learns by interacting with an environment and receiving feedback in the form of rewards or penalties.

# Applications of Machine Learning

Machine learning has numerous applications including image recognition, natural language processing, recommendation systems, and autonomous vehicles.
""",
        "sample_russian.txt": """
# Введение в машинное обучение

Машинное обучение — это подраздел искусственного интеллекта, который занимается разработкой алгоритмов, способных обучаться на данных и делать предсказания. Основные типы машинного обучения включают обучение с учителем, без учителя и с подкреплением.

## Обучение с учителем

Обучение с учителем предполагает тренировку модели на размеченных данных. Модель учится на примерах, которые содержат как входные данные, так и правильный выход. Распространенные алгоритмы включают линейную регрессию и метод опорных векторов.

## Обучение без учителя

Обучение без учителя находит закономерности в немаркированных данных. Модель пытается изучить структуру данных без явного руководства. Распространенные алгоритмы включают k-средних и метод главных компонент.

## Обучение с подкреплением

Обучение с подкреплением использует систему наград и наказаний для тренировки моделей. Агент учится, взаимодействуя со средой и получая обратную связь в виде вознаграждений или штрафов.

# Применение машинного обучения

Машинное обучение имеет множество применений, включая распознавание изображений, обработку естественного языка, рекомендательные системы и автономные транспортные средства.
""",
        "sample_german.txt": """
# Einführung in maschinelles Lernen

Maschinelles Lernen ist ein Teilbereich der künstlichen Intelligenz, der sich mit der Entwicklung von Algorithmen befasst, die aus Daten lernen und Vorhersagen treffen können. Die Haupttypen des maschinellen Lernens umfassen überwachtes Lernen, unüberwachtes Lernen und bestärkendes Lernen.

## Überwachtes Lernen

Beim überwachten Lernen wird ein Modell anhand von beschrifteten Daten trainiert. Das Modell lernt aus Beispielen, die sowohl Eingabedaten als auch die korrekte Ausgabe enthalten. Zu den gängigen Algorithmen gehören lineare Regression und Support Vector Machines.

## Unüberwachtes Lernen

Unüberwachtes Lernen findet Muster in unbeschrifteten Daten. Das Modell versucht, die Struktur der Daten ohne explizite Anleitung zu erlernen. Zu den gängigen Algorithmen gehören k-Means-Clustering und Hauptkomponentenanalyse.

## Bestärkendes Lernen

Bestärkendes Lernen verwendet ein System von Belohnungen und Bestrafungen, um Modelle zu trainieren. Der Agent lernt, indem er mit einer Umgebung interagiert und Feedback in Form von Belohnungen oder Strafen erhält.

# Anwendungen des maschinellen Lernens

Maschinelles Lernen hat zahlreiche Anwendungen, einschließlich Bilderkennung, Verarbeitung natürlicher Sprache, Empfehlungssysteme und autonome Fahrzeuge.
""",
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename, content in samples.items():
        file_path = output_dir / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content.strip())

    print(f"Созданы примеры учебных материалов в директории: {output_dir}")
    for filename in samples.keys():
        print(f"  - {filename}")


def main():
    """Основная функция CLI."""
    # Парсинг аргументов
    parser = setup_argparse()
    args = parser.parse_args()

    # Утилитарные команды
    if args.generate_config:
        create_config_file(args.config, overwrite=True)
        print(f"Конфигурационный файл создан: {args.config}")
        return 0

    if args.create_sample:
        create_sample_materials()
        return 0

    if args.list_languages:
        print("Поддерживаемые языки:")
        print("  - en / english (Английский)")
        print("  - ru / russian (Русский)")
        print("  - de / german (Немецкий)")
        print("\nДля автоопределения используйте: --language auto")
        return 0

    # Проверка обязательных параметров
    if not args.input and not args.text and not args.batch:
        parser.print_help()
        print("\nОшибка: необходимо указать --input, --text или --batch")
        return 1

    # Настройка логирования
    setup_logging(log_level=args.log_level, log_file=args.log_file, json_format=False)

    logger = logging.getLogger(__name__)

    # Загрузка конфигурации
    config = load_config(args.config)

    # Создание суммаризатора
    try:
        summarizer = create_summarizer(
            device=args.device if args.device != "auto" else None,
            use_advanced_detector=config.use_advanced_detector,
            cache_models=config.cache_models,
        )
    except Exception as e:
        logger.error(f"Ошибка инициализации суммаризатора: {e}")
        return 1

    # Обработка в зависимости от режима
    if args.batch:
        # Пакетная обработка
        output_dir = args.output or "outputs"
        stats = process_batch(
            input_dir=args.batch,
            output_dir=output_dir,
            language=args.language,
            compression=args.compression,
            abstractive=args.abstractive,
            extract_key_points=args.key_points,
            summarizer=summarizer,
            logger=logger,
        )

        print(f"\nПакетная обработка завершена:")
        print(f"  Всего файлов: {stats.get('total', 0)}")
        print(f"  Обработано: {stats.get('processed', 0)}")
        print(f"  С ошибками: {stats.get('failed', 0)}")
        print(f"  Пропущено: {stats.get('skipped', 0)}")
        print(f"\nРезультаты сохранены в: {output_dir}")

    elif args.input:
        # Обработка одиночного файла
        output_path = args.output

        # Если выходной файл не указан, генерируем имя
        if not output_path:
            input_path = Path(args.input)
            output_path = input_path.with_name(f"{input_path.stem}_summary.txt")

        success = process_single_file(
            input_path=args.input,
            output_path=output_path,
            language=args.language,
            compression=args.compression,
            abstractive=args.abstractive,
            extract_key_points=args.key_points,
            summarizer=summarizer,
            logger=logger,
        )

        if not success:
            return 1

    elif args.text:
        # Обработка текста из командной строки
        try:
            result = summarizer.summarize(
                text=args.text,
                language=args.language,
                compression=args.compression,
                abstractive=args.abstractive,
                extract_key_points=args.key_points,
            )

            output = format_result(result, "Входной текст")

            if args.output:
                save_text_file(output, args.output, overwrite=True)
                print(f"Результат сохранен в: {args.output}")
            else:
                print(output)

        except Exception as e:
            logger.error(f"Ошибка обработки текста: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

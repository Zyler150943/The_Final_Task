# Использование Multilingual Summarizer

## Репозиторий проекта
- GitHub: https://github.com/Zyler150943/The_Final_Task
- CI/CD Status: ![CI/CD](https://github.com/Zyler150943/The_Final_Task/actions/workflows/ci-cd.yml/badge.svg)

## Базовое использование

### Через командную строку:

```bash
# Клонирование репозитория
git clone https://github.com/Zyler150943/The_Final_Task.git
cd The_Final_Task

# Установка зависимостей
pip install -r requirements.txt

# Резюмирование файла на английском
python src/cli.py --input data/sample_english.txt --language en --compression 30

# Резюмирование текста напрямую на русском
python src/cli.py --text "Машинное обучение - это подраздел искусственного интеллекта..." --language ru --compression 20

# Пакетная обработка файлов
python src/cli.py --batch data/ --output results/ --language auto
```

### Через Python API:

```python
from src.core import create_summarizer

# Инициализация суммаризатора
summarizer = create_summarizer()

# Пример текста на русском
text = """
Машинное обучение — это подраздел искусственного интеллекта, который занимается разработкой алгоритмов, способных обучаться на данных и делать предсказания.
"""

# Резюмирование
result = summarizer.summarize(
    text=text,
    language='auto',  # автоопределение языка
    compression=30,   # уровень сжатия
    abstractive=True  # абстрактивное резюмирование
)

print(result.summary)
print(result.key_points)
```

## Поддерживаемые языки

- 🇬🇧 Английский (en)
- 🇷🇺 Русский (ru)
- 🇩🇪 Немецкий (de)

## Уровни сжатия

- 20% - максимальное сжатие
- 30% - рекомендуемый уровень (по умолчанию)
- 50% - минимальное сжатие

## Дополнительные возможности

### Пакетная обработка

Вы можете обработать несколько файлов одновременно:

```bash
# Обработка всех текстовых файлов в директории data/
python src/cli.py --batch data/ --output summaries/ --language auto --compression 30

# Обработка только английских файлов
python src/cli.py --batch data/english_docs/ --language en --compression 20
```

### Разные форматы вывода

По умолчанию результат выводится в консоль, но можно сохранить в файл:

```bash
# Сохранение в файл с метаданными
python src/cli.py --input document.txt --language en --compression 30 --output summary.txt

# Сохранение только резюме (без метаданных)
python src/cli.py --input document.txt --language ru --compression 20 --output summary_clean.txt --no-key-points
```

### Настройка конфигурации

Вы можете создать свой конфигурационный файл:

```bash
# Создание конфигурационного файла по умолчанию
python src/cli.py --generate-config my_config.json

# Использование своего конфигурационного файла
python src/cli.py --input text.txt --config my_config.json

# Использование конфигурации для разработки
python src/cli.py --input text.txt --config config.dev.json
```

## Примеры

### Пример 1: Резюмирование учебного материала

```bash
python src/cli.py --input data/sample_english.txt --language en --compression 30 --output summary_en.txt
```

Содержимое summary_en.txt:

```txt
=== Резюме: data/sample_english.txt ===
Язык: en
Уровень сжатия: 30%
Исходный размер: 1250 символов
Размер резюме: 375 символов
Коэффициент сжатия: 30.0%

=== Текст резюме ===
Machine learning is a subset of artificial intelligence focused on algorithms that learn from data. Main types include supervised learning (labeled data), unsupervised learning (pattern finding), and reinforcement learning (reward-based). Applications range from image recognition to natural language processing.

=== Ключевые моменты ===
1. Machine learning is an AI subset for data-driven algorithms
2. Three main types: supervised, unsupervised, reinforcement learning
3. Supervised learning uses labeled training data
4. Applications include image and language processing
5. Key steps: data collection, preprocessing, training, evaluation
```

### Пример 2: Быстрое резюмирование

```bash
python src/cli.py --text "Машинное обучение помогает компьютерам учиться на данных без явного программирования." --language ru --compression 50
```

Вывод в консоль:

```txt
Машинное обучение позволяет компьютерам обучаться на данных без программирования.
```

### Пример 3: Автоматическое определение языка

```bash
python src/cli.py --input data/sample_german.txt --language auto --compression 40
```

Вывод в консоль:

```txt
=== Резюме: data/sample_german.txt ===
Язык: de (автоопределен)
Уровень сжатия: 40%
Исходный размер: 980 символов
Размер резюме: 392 символов
Коэффициент сжатия: 40.0%

=== Текст резюме ===
Maschinelles Lernen ist ein Teilbereich der KI, bei dem Algorithmen aus Daten lernen. Haupttypen: überwachtes, unüberwachtes und bestärkendes Lernen. Anwendungen umfassen Bilderkennung und Sprachverarbeitung.
```

## Устранение неполадок

### Проблема: Модели не загружаются

Решение: Убедитесь, что у вас есть подключение к интернету для загрузки моделей. Вы можете предварительно скачать модели:

```bash
python scripts/download_models.py
```

Или используйте локальную копию моделей:

```bash
python src/cli.py --config config.dev.json --input text.txt
```

Или попробуйте использовать VPN, скорее всего, дело в геолокации.

### Проблема: Недостаточно памяти

Решение: Используйте меньшие модели или уменьшите размер входного текста:

```bash
# Использование CPU вместо GPU
python src/cli.py --device cpu --input text.txt

# Уменьшение максимальной длины текста
python src/cli.py --input text.txt --max-length 512
```

### Проблема: Не определяется язык

Решение: Укажите язык явно или используйте расширенный детектор:

```bash
# Явное указание языка
python src/cli.py --input text.txt --language en

# Использование расширенного детектора
python src/cli.py --input text.txt --language auto --use-advanced-detector
```

### Проблема: Ошибки кодировки

Решение: Укажите правильную кодировку файла:

```bash
# Для файлов в UTF-8
python src/cli.py --input text.txt --encoding utf-8

# Для файлов в Windows-1251
python src/cli.py --input text.txt --encoding cp1251
```

## Дополнительная помощь

Для получения дополнительной помощи используйте:

```bash
python src/cli.py --help
```

Или обратитесь к документации API.

## Интеграция с другими инструментами

### Использование в Jupyter Notebook:

```python
# В ячейке Jupyter
pip install -r requirements.txt

from src.core import create_summarizer

summarizer = create_summarizer()

# Резюмирование вывода ячейки
text = """
Длинный текст лекции или статьи...
"""

result = summarizer.summarize(text, language='auto', compression=30)
result.summary
```

### Использование в скриптах:

```python
#!/usr/bin/env python3
import sys
from src.core import create_summarizer
from src.utils.file_handler import load_text_file

def main():
    if len(sys.argv) < 2:
        print("Usage: script.py <input_file>")
        return
    
    input_file = sys.argv[1]
    text = load_text_file(input_file)
    
    summarizer = create_summarizer()
    result = summarizer.summarize(text, language='auto', compression=30)
    
    print(f"Summary of {input_file}:")
    print(result.summary)

if __name__ == "__main__":
    main()
```

Примечание: Для получения лучших результатов используйте тексты объемом от 100 до 5000 символов. Слишком короткие тексты могут давать некачественные резюме, а слишком длинные - требовать больше времени на обработку.
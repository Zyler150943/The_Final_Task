# API Documentation

## Обзор

Multilingual Summarizer предоставляет несколько способов интеграции:

1. **Командный интерфейс (CLI)** - для быстрого использования из терминала
2. **Python API** - для интеграции в Python приложения
3. **Модульная архитектура** - возможность использования отдельных компонентов

## TextSummarizer Class

Основной класс для резюмирования текста.

### Инициализация

```python
from src.core import create_summarizer

# Создание суммаризатора с параметрами по умолчанию
summarizer = create_summarizer()

# Создание с дополнительными параметрами
summarizer = create_summarizer(
    device='cpu',                # Использовать только CPU
    use_advanced_detector=True,  # Использовать расширенный детектор языка
    cache_models=True            # Кэшировать модели для повторного использования
)
```

Параметры инициализации:
- device (str): Устройство для вычислений ('cuda', 'cpu', 'auto')
- use_advanced_detector (bool): Использовать FastText для определения языка
- cache_models (bool): Кэшировать загруженные модели в памяти

### Основные методы

#### summarize(text, language='auto', compression=30, abstractive=True, extract_key_points=True)

Резюмирует текст и возвращает объект SummaryResult.

Параметры:
- text (str): Текст для резюмирования
- language (str): Язык текста ('en', 'ru', 'de', 'auto')
- compression (int): Уровень сжатия (20, 30, 50)
- abstractive (bool): Использовать абстрактивное резюмирование
- extract_key_points (bool): Извлекать ключевые моменты

Возвращает: SummaryResult объект

Пример использования:

```python
result = summarizer.summarize(
    text="Длинный текст для резюмирования...",
    language='auto',      # автоопределение языка
    compression=30,       # сжатие до 30%
    abstractive=True,     # абстрактивное резюмирование
    extract_key_points=True  # извлечь ключевые моменты
)

# Доступ к результатам
print(f"Резюме: {result.summary}")
print(f"Язык: {result.language}")
print(f"Ключевые моменты: {result.key_points}")
print(f"Коэффициент сжатия: {result.compression_ratio:.1%}")
```

#### summarize_batch(texts, language='auto', compression=30, batch_size=4, show_progress=True)

Пакетное резюмирование текстов.

Параметры:
- texts (List[str]): Список текстов для резюмирования
- language (str): Язык текстов
- compression (int): Уровень сжатия
- batch_size (int): Размер пакета для обработки
- show_progress (bool): Показывать прогресс-бар

Возвращает: Список объектов SummaryResult

Пример использования:

```python
texts = [
    "Первый текст для резюмирования...",
    "Второй текст на английском...",
    "Третий текст на немецком..."
]

results = summarizer.summarize_batch(
    texts=texts,
    language='auto',      # автоопределение для каждого текста
    compression=30,
    batch_size=2,
    show_progress=True
)

for i, result in enumerate(results):
    if result:
        print(f"Текст {i+1}: {result.summary[:50]}...")
```

#### detect_language(text, use_advanced=True)

Определяет язык текста.

Параметры:
- text (str): Текст для определения языка
- use_advanced (bool): Использовать расширенный детектор

Возвращает: Код языка ('en', 'ru', 'de') или 'unknown'

Пример использования:

```python
language = summarizer.detect_language("This is an English text.")
print(f"Определен язык: {language}")  # Вывод: en

language = summarizer.detect_language("Это русский текст.", use_advanced=False)
print(f"Определен язык: {language}")  # Вывод: ru
```

#### get_supported_languages()

Получает список поддерживаемых языков.

Возвращает: Список кодов поддерживаемых языков

Пример использования:

```python
languages = summarizer.get_supported_languages()
print(f"Поддерживаемые языки: {languages}")  # Вывод: ['en', 'ru', 'de']
```

#### get_compression_levels()

Получает доступные уровни сжатия.

Возвращает: Список доступных уровней сжатия

Пример использования:

```python
levels = summarizer.get_compression_levels()
print(f"Доступные уровни сжатия: {levels}")  # Вывод: [20, 30, 50]
```

#### clear_model_cache()

Очищает кэш моделей.

Пример использования:

```python
# Очистка кэша для освобождения памяти
summarizer.clear_model_cache()
```

## SummaryResult Dataclass

Содержит результаты резюмирования:

```python
@dataclass
class SummaryResult:
    original_text: str          # Исходный текст
    summary: str                # Резюмированный текст
    language: str               # Определенный язык
    compression: float          # Использованный уровень сжатия
    key_points: List[str]       # Список ключевых моментов
    original_length: int        # Длина исходного текста
    summary_length: int         # Длина резюме
    compression_ratio: float    # Коэффициент сжатия
    processing_time: Optional[float] = None  # Время обработки в секундах
```

Пример использования:

```python
result = summarizer.summarize(text, language='auto', compression=30)

print("=== Статистика ===")
print(f"Исходная длина: {result.original_length} символов")
print(f"Длина резюме: {result.summary_length} символов")
print(f"Коэффициент сжатия: {result.compression_ratio:.1%}")

if result.processing_time:
    print(f"Время обработки: {result.processing_time:.2f} секунд")

print("\n=== Резюме ===")
print(result.summary)

if result.key_points:
    print("\n=== Ключевые моменты ===")
    for i, point in enumerate(result.key_points, 1):
        print(f"{i}. {point}")
```

## LanguageDetector Class

Класс для определения языка текста.

### Инициализация

```python
from src.core import create_language_detector

# Создание детектора с параметрами по умолчанию
detector = create_language_detector()

# Создание с расширенными возможностями
detector = create_language_detector(
    use_fasttext=True,      # Использовать FastText
    model_path='lid.176.ftz'  # Путь к модели FastText
)
```

### Основные методы

#### detect_language(text, prefer_fasttext=None)

Определяет язык текста.

Параметры:
- text (str): Текст для определения языка
- prefer_fasttext (bool): Предпочитать FastText для определения

Возвращает: Словарь с результатами

Пример использования:

```python
result = detector.detect_language("This is an English text.")

print(f"Код языка: {result['language_code']}")        # en
print(f"Название языка: {result['language_name']}")   # english
print(f"Уверенность: {result['confidence']:.2f}")     # 0.95
print(f"Метод определения: {result['method']}")       # langdetect или fasttext
print(f"Поддерживается: {result['is_supported']}")    # True
```

#### detect_language_batch(texts, prefer_fasttext=None)

Пакетное определение языка.

Параметры:
- texts (List[str]): Список текстов
- prefer_fasttext (bool): Предпочитать FastText

Возвращает: Список результатов

Пример использования:

```python
texts = ["English text", "Русский текст", "Deutscher Text"]
results = detector.detect_language_batch(texts)

for i, result in enumerate(results):
    print(f"Текст {i+1}: {result['language_code']}")
```

#### is_language_supported(language_code)

Проверяет поддержку языка.

Параметры:
- language_code (str): Код языка

Возвращает: True если язык поддерживается

Пример использования:

```python
if detector.is_language_supported('en'):
    print("Английский язык поддерживается")
```

#### get_language_name(language_code)

Получает название языка по коду.

Параметры:
- language_code (str): Код языка

Возвращает: Название языка или None

Пример использования:

```python
name = detector.get_language_name('ru')
print(name)  # russian
```

#### get_available_methods()

Получает список доступных методов определения языка.

Возвращает: Список доступных методов

Пример использования:

```python
methods = detector.get_available_methods()
print(f"Доступные методы: {methods}")  # ['langdetect', 'fasttext']
```
## TextProcessor Class

Класс для предобработки текста.

### Инициализация

```python
from src.core import create_text_processor

# Создание процессора для английского языка
processor = create_text_processor(language='en')

# Создание процессора для русского языка
processor_ru = create_text_processor(language='ru')
```

### Основные методы

#### clean_text(text, remove_stopwords=False)

Очищает текст от лишних символов и нормализует его.

Параметры:
- text (str): Исходный текст
- remove_stopwords (bool): Удалять стоп-слова

Возвращает: Очищенный текст

Пример использования:

```python
text = "This is a sample text with https://example.com and email@example.com"
cleaned = processor.clean_text(text)
print(cleaned)  # "This is a sample text with and"
```

#### split_into_sentences(text)

Разбивает текст на предложения.

Параметры:
- text (str): Исходный текст

Возвращает: Список предложений

Пример использования:

```python
text = "First sentence. Second sentence! Third sentence?"
sentences = processor.split_into_sentences(text)
print(sentences)  # ['First sentence.', 'Second sentence!', 'Third sentence?']
```

#### split_into_paragraphs(text)

Разбивает текст на абзацы.

Параметры:
- text (str): Исходный текст

Возвращает: Список абзацев

Пример использования:

```python
text = "First paragraph\n\nSecond paragraph\n\nThird paragraph"
paragraphs = processor.split_into_paragraphs(text)
print(len(paragraphs))  # 3
```

#### calculate_text_statistics(text)

Рассчитывает статистику текста.

Параметры:
- text (str): Исходный текст

Возвращает: Словарь со статистикой

Пример использования:

```python
stats = processor.calculate_text_statistics("This is a sample text.")
print(f"Всего слов: {stats['total_words']}")
print(f"Всего предложений: {stats['total_sentences']}")
print(f"Уникальных слов: {stats['unique_words']}")
print(f"Средняя длина предложения: {stats['avg_sentence_length']:.2f}")
```

#### extract_key_phrases(text, top_n=10)

Извлекает ключевые фразы из текста.

Параметры:
- text (str): Исходный текст
- top_n (int): Количество извлекаемых фраз

Возвращает: Список кортежей (фраза, вес)

Пример использования:

```python
text = "Machine learning is a subset of artificial intelligence."
phrases = processor.extract_key_phrases(text, top_n=3)
for phrase, weight in phrases:
    print(f"{phrase}: {weight:.3f}")
```

#### split_long_text(text, max_tokens=512)

Разделяет длинный текст на части.

Параметры:
- text (str): Исходный текст
- max_tokens (int): Максимальное количество токенов в части

Возвращает: Список частей текста

Пример использования:

```python
long_text = "Very long text..." * 100
parts = processor.split_long_text(long_text, max_tokens=256)
print(f"Текст разделен на {len(parts)} частей")
```

#### prepare_for_summarization(text, max_length=None)

Подготавливает текст для резюмирования.

Параметры:
- text (str): Исходный текст
- max_length (int): Максимальная длина текста

Возвращает: Подготовленный текст

Пример использования:

```python
text = "Unprepared text with URLs and emails."
prepared = processor.prepare_for_summarization(text, max_length=1000)
print(prepared)
```

#### detect_language_from_text(text)

Простое определение языка по символам.

Параметры:
- text (str): Исходный текст

Возвращает: Предполагаемый код языка

Пример использования:

```python
language = processor.detect_language_from_text("Это русский текст")
print(language)  # ru
```

## ModelFactory

Фабрика для создания моделей резюмирования.

```python
from src.models.abstractive import ModelFactory

# Создание модели для английского языка
english_model = ModelFactory.create_model('en', device='cpu')

# Создание модели для русского языка с кастомным именем
russian_model = ModelFactory.create_model('ru', model_name='custom-model', device='cpu')

# Получение списка поддерживаемых языков
languages = ModelFactory.get_available_languages()
print(languages)  # ['en', 'ru', 'de']

# Получение имени модели по умолчанию для языка
model_name = ModelFactory.get_default_model_name('de')
print(model_name)  # 'ml6team/mt5-small-german-finetune-mlsum'
```

## AbstractiveModel

Базовый класс для абстрактивных моделей.

```python
from src.models.abstractive import EnglishSummarizer, RussianSummarizer, GermanSummarizer

# Создание специализированных моделей
english_model = EnglishSummarizer(device='cpu')
russian_model = RussianSummarizer(device='cpu')
german_model = GermanSummarizer(device='cpu')

# Загрузка модели
english_model.load()

# Резюмирование
summary = english_model.summarize(
    text="Long English text...",
    max_length=150,
    min_length=30,
    temperature=1.0,
    num_beams=4
)
```

## BatchSummarizer

Класс для пакетного резюмирования текстов.

```python
from src.models.abstractive import BatchSummarizer

# Создание пакетного суммаризатора
batch_processor = BatchSummarizer(english_model, batch_size=4)

# Пакетное резюмирование
texts = ["Text 1...", "Text 2...", "Text 3..."]
summaries = batch_processor.summarize_batch(
    texts,
    max_length=100,
    min_length=20,
    show_progress=True
)
```

## Пример полного сценария использования

```python
from src.core import create_summarizer, create_text_processor
from src.utils.file_handler import load_text_file, save_text_file
from src.utils.logger import setup_logging

# Настройка логирования
setup_logging(log_level='INFO')

# Загрузка текста из файла
text = load_text_file('input.txt')

# Предобработка текста
processor = create_text_processor(language='auto')
cleaned_text = processor.clean_text(text)
stats = processor.calculate_text_statistics(cleaned_text)
print(f"Статистика текста: {stats}")

# Создание суммаризатора
summarizer = create_summarizer(
    device='auto',
    use_advanced_detector=True,
    cache_models=True
)

# Определение языка
language = summarizer.detect_language(cleaned_text)
print(f"Определен язык: {language}")

# Резюмирование
result = summarizer.summarize(
    text=cleaned_text,
    language=language,
    compression=30,
    abstractive=True,
    extract_key_points=True
)

# Вывод результатов
print(f"\n=== Результаты резюмирования ===")
print(f"Исходный размер: {result.original_length} символов")
print(f"Размер резюме: {result.summary_length} символов")
print(f"Коэффициент сжатия: {result.compression_ratio:.1%}")
print(f"Время обработки: {result.processing_time:.2f} секунд")

print(f"\n=== Резюме ===")
print(result.summary)

print(f"\n=== Ключевые моменты ===")
for i, point in enumerate(result.key_points, 1):
    print(f"{i}. {point}")

# Сохранение результатов
save_text_file(result.summary, 'output.txt')
save_text_file('\n'.join(result.key_points), 'key_points.txt')

# Сохранение отчета
report = f"""Отчет о резюмировании:
Дата: {datetime.now()}
Исходный файл: input.txt
Язык: {result.language}
Уровень сжатия: {result.compression}%
Исходный размер: {result.original_length}
Размер резюме: {result.summary_length}
Коэффициент сжатия: {result.compression_ratio:.1%}
Время обработки: {result.processing_time:.2f} секунд

Резюме:
{result.summary}

Ключевые моменты:
{chr(10).join(result.key_points)}
"""

save_text_file(report, 'report.txt')
```

Примечание: Все примеры кода предполагают, что зависимости установлены и модели загружены. Для первого запуска может потребоваться некоторое время для загрузки моделей из интернета.
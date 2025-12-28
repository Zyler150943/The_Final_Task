# -*- coding: utf-8 -*-
"""
Модуль для управления конфигурацией приложения.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict, field

# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Конфигурация модели для конкретного языка."""
    
    language: str
    model_name: str
    max_input_length: int = 1024
    max_output_length: int = 512
    min_output_length: int = 30
    temperature: float = 1.0
    num_beams: int = 4
    do_sample: bool = False
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0


@dataclass
class TextProcessingConfig:
    """Конфигурация обработки текста."""
    
    clean_text: bool = True
    remove_stopwords: bool = False
    remove_punctuation: bool = False
    to_lowercase: bool = False
    max_text_length: int = 50000
    min_text_length: int = 10
    split_long_texts: bool = True
    max_tokens_per_chunk: int = 512
    preserve_paragraphs: bool = True


@dataclass
class SummarizationConfig:
    """Конфигурация резюмирования."""
    
    compression_levels: list = field(default_factory=lambda: [20, 30, 50])
    default_compression: int = 30
    extract_key_points: bool = True
    key_points_count: int = 5
    abstractive: bool = True
    batch_size: int = 4
    show_progress: bool = True
    save_statistics: bool = True


@dataclass
class LanguageDetectionConfig:
    """Конфигурация определения языка."""
    
    min_text_length: int = 10
    confidence_threshold: float = 0.6
    fallback_language: str = "en"
    use_fasttext: bool = False
    fasttext_model_path: Optional[str] = None


@dataclass
class PathConfig:
    """Конфигурация путей."""
    
    data_dir: str = "data"
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    cache_dir: str = ".cache"
    config_dir: str = "configs"
    temp_dir: str = "temp"


@dataclass
class FileHandlingConfig:
    """Конфигурация работы с файлами."""
    
    default_encoding: str = "utf-8"
    auto_detect_encoding: bool = True
    supported_extensions: list = field(
        default_factory=lambda: [".txt", ".md", ".rst", ".json", ".yaml", ".yml"]
    )
    max_file_size_mb: int = 10
    backup_files: bool = True
    overwrite_output: bool = False


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    
    console_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    json_format: bool = False
    rotate_logs: bool = True
    max_log_size_mb: int = 10
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Конфигурация производительности."""
    
    use_caching: bool = True
    cache_expiry_hours: int = 24
    max_cached_items: int = 100
    preload_models: bool = False
    parallel_processing: bool = True
    max_concurrent_tasks: int = 4


@dataclass
class OutputConfig:
    """Конфигурация вывода."""
    
    include_metadata: bool = True
    include_statistics: bool = True
    include_key_points: bool = True
    format: str = "text"
    timestamp_format: str = "%Y-%m-%d_%H-%M-%S"
    output_encoding: str = "utf-8"


@dataclass
class SecurityConfig:
    """Конфигурация безопасности."""
    
    encrypt_config: bool = False
    encryption_key: str = ""
    validate_input: bool = True
    max_input_length: int = 100000
    allowed_file_types: list = field(default_factory=lambda: [".txt", ".md", ".rst"])


@dataclass
class Config:
    """Основная конфигурация приложения."""
    
    # Основные настройки
    project_name: str = "Multilingual Learning Material Summarizer"
    version: str = "1.0.0"
    debug: bool = False
    
    # Системные настройки
    device: str = "auto"
    cache_models: bool = True
    log_level: str = "INFO"
    max_workers: int = 2
    
    # Языковые настройки
    supported_languages: list = field(default_factory=lambda: ['en', 'ru', 'de'])
    default_language: str = 'en'
    auto_detect_language: bool = True
    use_advanced_detector: bool = False
    
    # Конфигурации компонентов
    model_configs: Dict[str, ModelConfig] = field(default_factory=dict)
    text_processing: TextProcessingConfig = field(default_factory=TextProcessingConfig)
    summarization: SummarizationConfig = field(default_factory=SummarizationConfig)
    language_detection: LanguageDetectionConfig = field(
        default_factory=LanguageDetectionConfig
    )
    paths: PathConfig = field(default_factory=PathConfig)
    file_handling: FileHandlingConfig = field(default_factory=FileHandlingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Дополнительные секции (опциональные)
    api: Optional[Dict[str, Any]] = None
    monitoring: Optional[Dict[str, Any]] = None
    notifications: Optional[Dict[str, Any]] = None
    scheduled_tasks: Optional[Dict[str, Any]] = None
    testing: Optional[Dict[str, Any]] = None
    updates: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Инициализация после создания объекта."""
        # Инициализируем конфигурации моделей по умолчанию
        if not self.model_configs:
            self._init_default_model_configs()
        
        # Создаем директории если нужно
        self._create_directories()
        
        # Валидация конфигурации
        self._validate_config()
    
    def _init_default_model_configs(self):
        """Инициализация конфигураций моделей по умолчанию."""
        default_configs = {
            'en': ModelConfig(
                language='en',
                model_name='facebook/bart-large-cnn',
                max_input_length=1024,
                max_output_length=512,
                min_output_length=30
            ),
            'ru': ModelConfig(
                language='ru',
                model_name='cointegrated/rut5-base-absum',
                max_input_length=512,
                max_output_length=200,
                min_output_length=20
            ),
            'de': ModelConfig(
                language='de',
                model_name='ml6team/mt5-small-german-finetune-mlsum',
                max_input_length=512,
                max_output_length=200,
                min_output_length=20
            )
        }
        self.model_configs = default_configs
    
    def _create_directories(self):
        """Создание необходимых директорий."""
        dirs = [
            self.paths.data_dir,
            self.paths.outputs_dir,
            self.paths.logs_dir,
            self.paths.cache_dir,
            self.paths.config_dir,
            self.paths.temp_dir
        ]
        
        for directory in dirs:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Не удалось создать директорию {directory}: {e}")
    
    def _validate_config(self):
        """Валидация конфигурации."""
        # Проверка поддерживаемых языков
        for lang in self.supported_languages:
            if lang not in self.model_configs:
                logger.warning(f"Для языка {lang} нет конфигурации модели")
        
        # Проверка уровней сжатия
        for level in self.summarization.compression_levels:
            if level not in [20, 30, 50]:
                logger.warning(f"Неподдерживаемый уровень сжатия: {level}")
        
        # Проверка максимальной длины текста
        if self.security.max_input_length < self.text_processing.max_text_length:
            logger.warning(
                f"Максимальная длина текста для обработки ({self.text_processing.max_text_length}) "
                f"превышает максимальную длину ввода ({self.security.max_input_length})"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразование конфигурации в словарь."""
        result = {}
        
        for key, value in self.__dict__.items():
            if hasattr(value, '__dict__') or isinstance(value, dict):
                # Рекурсивно преобразуем объекты и словари
                if hasattr(value, '__dict__'):
                    result[key] = asdict(value)
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Сериализация конфигурации в JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save(self, file_path: str):
        """
        Сохранение конфигурации в файл.
        
        Args:
            file_path: Путь к файлу
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(self.to_json())
            
            logger.info(f"Конфигурация сохранена в {file_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения конфигурации: {e}")
            raise
    
    def update(self, config_dict: Dict[str, Any]):
        """
        Обновление конфигурации из словаря.
        
        Args:
            config_dict: Словарь с новыми значениями
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                
                if isinstance(current_value, dict) and isinstance(value, dict):
                    # Рекурсивное обновление словарей
                    current_value.update(value)
                elif hasattr(current_value, '__dict__') and isinstance(value, dict):
                    # Обновление объектов dataclass
                    for sub_key, sub_value in value.items():
                        if hasattr(current_value, sub_key):
                            setattr(current_value, sub_key, sub_value)
                else:
                    setattr(self, key, value)
            else:
                logger.debug(f"Неизвестный параметр конфигурации: {key}")
    
    def get_model_config(self, language: str) -> Optional[ModelConfig]:
        """
        Получение конфигурации модели для языка.
        
        Args:
            language: Код языка
        
        Returns:
            Конфигурация модели или None
        """
        return self.model_configs.get(language)
    
    def is_language_supported(self, language: str) -> bool:
        """
        Проверка поддержки языка.
        
        Args:
            language: Код языка
        
        Returns:
            True если язык поддерживается
        """
        return language in self.supported_languages
    
    def get_default_model_name(self, language: str) -> Optional[str]:
        """
        Получение имени модели по умолчанию для языка.
        
        Args:
            language: Код языка
        
        Returns:
            Имя модели или None
        """
        config = self.get_model_config(language)
        return config.model_name if config else None


def load_config(file_path: Optional[str] = None) -> Config:
    """
    Загрузка конфигурации из файла или создание по умолчанию.
    
    Args:
        file_path: Путь к файлу конфигурации
    
    Returns:
        Объект конфигурации
    """
    # Создаем конфигурацию по умолчанию
    config = Config()
    
    if file_path:
        file_path = Path(file_path)
        
        # Если файл существует, загружаем его
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    config_dict = json.load(file)
                
                config.update(config_dict)
                logger.info(f"Конфигурация загружена из {file_path}")
                
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка парсинга JSON в файле {file_path}: {e}")
                raise
            except Exception as e:
                logger.error(f"Ошибка загрузки конфигурации из {file_path}: {e}")
                raise
        
        # Если файла нет, создаем его с конфигурацией по умолчанию
        else:
            logger.warning(f"Файл конфигурации {file_path} не найден, создаем новый")
            config.save(str(file_path))
    
    return config


def save_config(config: Config, file_path: str):
    """
    Сохранение конфигурации в файл.
    
    Args:
        config: Объект конфигурации
        file_path: Путь к файлу
    """
    config.save(file_path)


def get_default_config() -> Config:
    """
    Получение конфигурации по умолчанию.
    
    Returns:
        Конфигурация по умолчанию
    """
    return Config()


def create_config_file(file_path: str, overwrite: bool = False):
    """
    Создание файла конфигурации с настройками по умолчанию.
    
    Args:
        file_path: Путь к файлу
        overwrite: Перезаписывать существующий файл
    """
    file_path = Path(file_path)
    
    if file_path.exists() and not overwrite:
        logger.warning(f"Файл уже существует: {file_path}")
        return
    
    config = get_default_config()
    config.save(str(file_path))
    logger.info(f"Создан файл конфигурации: {file_path}")


def get_config_path(config_name: str = "config.json") -> Path:
    """
    Получение пути к конфигурационному файлу.
    
    Args:
        config_name: Имя конфигурационного файла
    
    Returns:
        Путь к файлу конфигурации
    """
    # Ищем файл в разных местах
    possible_paths = [
        Path(config_name),  # Текущая директория
        Path("configs") / config_name,  # Папка configs
        Path.home() / ".multilingual_summarizer" / config_name,  # Домашняя директория
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    # Если файл не найден, возвращаем путь в текущей директории
    return Path(config_name)
"""
Модуль utils содержит вспомогательные функции и утилиты.
"""

from .file_handler import (
    load_text_file,
    save_text_file,
    load_config_file,
    detect_file_encoding,
    get_file_info,
)

from .config import Config, load_config, save_config, get_default_config

from .logger import setup_logging, get_logger
from .helpers import (
    format_size,
    calculate_compression_ratio,
    validate_text,
    timeit_decorator,
)

__all__ = [
    # file_handler
    "load_text_file",
    "save_text_file",
    "load_config_file",
    "detect_file_encoding",
    "get_file_info",
    # config
    "Config",
    "load_config",
    "save_config",
    "get_default_config",
    # logger
    "setup_logging",
    "get_logger",
    # helpers
    "format_size",
    "calculate_compression_ratio",
    "validate_text",
    "timeit_decorator",
]

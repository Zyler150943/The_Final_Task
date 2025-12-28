"""
Модуль utils содержит вспомогательные функции и утилиты.
"""

from .config import Config, get_default_config, load_config, save_config
from .file_handler import (
    detect_file_encoding,
    get_file_info,
    load_config_file,
    load_text_file,
    save_text_file,
)
from .helpers import (
    calculate_compression_ratio,
    format_size,
    timeit_decorator,
    validate_text,
)
from .logger import get_logger, setup_logging

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
# -*- coding: utf-8 -*-
"""
Модуль для настройки логирования.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Форматтер для вывода логов в JSON формате."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Форматирование записи лога в JSON."""
        log_object = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Добавляем дополнительные поля
        if hasattr(record, 'extra'):
            log_object.update(record.extra)
        
        # Добавляем информацию об исключении
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_object, ensure_ascii=False)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Настройка логирования для приложения.
    
    Args:
        log_level: Уровень логирования
        log_file: Путь к файлу логов
        json_format: Использовать JSON формат
        log_dir: Директория для логов
    
    Returns:
        Настроенный логгер
    """
    # Создаем директорию для логов
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Устанавливаем уровень логирования
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Создаем корневой логгер
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Удаляем существующие обработчики
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Создаем обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        console_handler.setFormatter(logging.Formatter(console_format))
    
    logger.addHandler(console_handler)
    
    # Создаем обработчик для файла если указан
    if log_file:
        # Если указано только имя файла, добавляем путь к директории логов
        if not Path(log_file).is_absolute():
            log_file = log_dir_path / log_file
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            file_handler.setFormatter(logging.Formatter(file_format))
        
        logger.addHandler(file_handler)
    
    # Настройка логирования для сторонних библиотек
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('nltk').setLevel(logging.WARNING)
    
    logger.info(f"Логирование настроено, уровень: {log_level}")
    
    return logger


def get_logger(name: str, extra: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Получение именованного логгера с дополнительными полями.
    
    Args:
        name: Имя логгера
        extra: Дополнительные поля для логов
    
    Returns:
        Именованный логгер
    """
    logger = logging.getLogger(name)
    
    if extra:
        # Добавляем дополнительные поля к записям лога
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            record.extra = extra
            return record
        
        logging.setLogRecordFactory(record_factory)
    
    return logger


def log_execution_time(logger: logging.Logger):
    """
    Декоратор для логирования времени выполнения функций.
    
    Args:
        logger: Логгер для записи
    
    Returns:
        Декоратор функции
    """
    def decorator(func):
        import time
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                logger.debug(
                    f"Функция {func.__name__} выполнена за {execution_time:.2f} секунд",
                    extra={'execution_time': execution_time}
                )
                
                return result
            
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Функция {func.__name__} завершилась с ошибкой "
                    f"за {execution_time:.2f} секунд: {e}",
                    extra={'execution_time': execution_time, 'error': str(e)}
                )
                raise
        
        return wrapper
    
    return decorator


class LoggingContext:
    """
    Контекстный менеджер для временного изменения уровня логирования.
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        handler: Optional[logging.Handler] = None,
        close: bool = True
    ):
        """
        Инициализация контекста логирования.
        
        Args:
            logger: Логгер
            level: Временный уровень логирования
            handler: Временный обработчик
            close: Закрывать обработчик при выходе
        """
        self.logger = logger
        self.level = level
        self.handler = handler
        self.close = close
        
        self.old_level = None
        self.added_handler = False
    
    def __enter__(self):
        """Вход в контекст."""
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        
        if self.handler:
            self.logger.addHandler(self.handler)
            self.added_handler = True
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Выход из контекста."""
        if self.level is not None and self.old_level is not None:
            self.logger.setLevel(self.old_level)
        
        if self.handler and self.added_handler:
            self.logger.removeHandler(self.handler)
        
        if self.handler and self.close:
            self.handler.close()
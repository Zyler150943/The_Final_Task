#!/usr/bin/env python3
"""
Скрипт для предварительной загрузки моделей.
"""

import logging
from pathlib import Path

from src.models.abstractive import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_models(languages=None, device='cpu'):
    """
    Загрузка моделей для указанных языков.
    
    Args:
        languages: Список языков для загрузки (по умолчанию все)
        device: Устройство для загрузки моделей
    """
    if languages is None:
        languages = ['en', 'ru', 'de']
    
    for lang in languages:
        try:
            logger.info(f"Downloading model for language: {lang}")
            model = ModelFactory.create_model(lang, device=device)
            model.load()
            logger.info(f"Model for {lang} downloaded successfully")
        except Exception as e:
            logger.error(f"Failed to download model for {lang}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download NLP models for summarization')
    parser.add_argument('--languages', nargs='+', default=['en', 'ru', 'de'],
                       help='Languages to download models for')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                       help='Device to load models on')
    
    args = parser.parse_args()
    
    download_models(args.languages, args.device)
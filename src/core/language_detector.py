# -*- coding: utf-8 -*-
"""
Модуль для определения языка текста.
"""

import logging
from typing import Dict, Optional, Tuple

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import fasttext
from fasttext.FastText import _FastText

# Настройка детерминизма для langdetect
DetectorFactory.seed = 0

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LanguageDetector:
    """
    Класс для определения языка текста.
    Поддерживает английский, русский и немецкий языки.
    """

    # Коды языков и их соответствие
    LANGUAGE_CODES = {
        "en": "english",
        "ru": "russian",
        "de": "german",
        "english": "en",
        "russian": "ru",
        "german": "de",
    }

    # Порог уверенности для определения языка
    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, use_fasttext: bool = False, model_path: Optional[str] = None):
        """
        Инициализация детектора языка.

        Args:
            use_fasttext: Использовать FastText для определения языка
            model_path: Путь к модели FastText (если используется)
        """
        self.use_fasttext = use_fasttext
        self.fasttext_model = None

        if use_fasttext:
            self._load_fasttext_model(model_path)

    def _load_fasttext_model(self, model_path: Optional[str] = None):
        """
        Загрузка модели FastText.

        Args:
            model_path: Путь к модели FastText
        """
        try:
            if model_path:
                self.fasttext_model = fasttext.load_model(model_path)
            else:
                # Попробуем загрузить предварительно обученную модель
                self.fasttext_model = fasttext.load_model("lid.176.ftz")
            logger.info("Модель FastText успешно загружена")
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель FastText: {e}")
            logger.info("Будет использован langdetect для определения языка")
            self.use_fasttext = False

    def detect_with_langdetect(self, text: str) -> Tuple[Optional[str], float]:
        """
        Определение языка с использованием langdetect.

        Args:
            text: Текст для определения языка

        Returns:
            Кортеж (код языка, уверенность)
        """
        try:
            # Берем достаточный объем текста для точного определения
            sample_text = text[:1000] if len(text) > 1000 else text

            if len(sample_text.strip()) < 10:
                return None, 0.0

            detected = detect(sample_text)
            confidence = 1.0  # langdetect не предоставляет уверенность

            # Преобразуем в наш формат
            if detected in self.LANGUAGE_CODES:
                return self.LANGUAGE_CODES[detected], confidence
            else:
                logger.warning(f"Обнаружен неподдерживаемый язык: {detected}")
                return None, 0.0

        except LangDetectException as e:
            logger.error(f"Ошибка определения языка с langdetect: {e}")
            return None, 0.0
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при определении языка: {e}")
            return None, 0.0

    def detect_with_fasttext(self, text: str) -> Tuple[Optional[str], float]:
        """
        Определение языка с использованием FastText.

        Args:
            text: Текст для определения языка

        Returns:
            Кортеж (код языка, уверенность)
        """
        if not self.fasttext_model:
            return None, 0.0

        try:
            # FastText требует минимальную длину текста
            if len(text.strip()) < 10:
                return None, 0.0

            # Предсказываем язык
            predictions = self.fasttext_model.predict(text, k=1)

            if predictions and len(predictions[0]) > 0:
                label = predictions[0][0]
                confidence = float(predictions[1][0])

                # Извлекаем код языка из метки (например, '__label__en')
                if label.startswith("__label__"):
                    lang_code = label[9:11]  # Берем первые 2 символа после __label__

                    if lang_code in self.LANGUAGE_CODES:
                        return self.LANGUAGE_CODES[lang_code], confidence
                    else:
                        logger.warning(
                            f"FastText обнаружил неподдерживаемый язык: {lang_code}"
                        )
                        return None, 0.0

            return None, 0.0

        except Exception as e:
            logger.error(f"Ошибка определения языка с FastText: {e}")
            return None, 0.0

    def detect_language(
        self, text: str, prefer_fasttext: Optional[bool] = None
    ) -> Dict[str, any]:
        """
        Определение языка текста.

        Args:
            text: Текст для определения языка
            prefer_fasttext: Предпочитать FastText для определения
                           (если None, используется настройка из инициализации)

        Returns:
            Словарь с результатами:
            {
                'language_code': 'en'/'ru'/'de',
                'language_name': 'english'/'russian'/'german',
                'confidence': 0.0-1.0,
                'method': 'langdetect'/'fasttext',
                'is_supported': bool
            }
        """
        if not text or len(text.strip()) < 10:
            return {
                "language_code": None,
                "language_name": None,
                "confidence": 0.0,
                "method": "none",
                "is_supported": False,
                "error": "Текст слишком короткий для определения языка",
            }

        use_fasttext = (
            prefer_fasttext if prefer_fasttext is not None else self.use_fasttext
        )

        # Определяем язык выбранным методом
        if use_fasttext and self.fasttext_model:
            language_code, confidence = self.detect_with_fasttext(text)
            method = "fasttext"
        else:
            language_code, confidence = self.detect_with_langdetect(text)
            method = "langdetect"

        # Проверяем, поддерживается ли язык
        is_supported = language_code in ["en", "ru", "de"]

        # Получаем название языка
        language_name = (
            self.LANGUAGE_CODES.get(language_code) if language_code else None
        )

        result = {
            "language_code": language_code,
            "language_name": language_name,
            "confidence": confidence,
            "method": method,
            "is_supported": is_supported,
            "error": None,
        }

        # Логируем результат
        if language_code:
            logger.info(
                f"Определен язык: {language_name} ({language_code}), "
                f"уверенность: {confidence:.2f}, метод: {method}"
            )
        else:
            logger.warning(f"Не удалось определить язык текста, метод: {method}")

        return result

    def detect_language_batch(
        self, texts: list, prefer_fasttext: Optional[bool] = None
    ) -> list:
        """
        Пакетное определение языка для нескольких текстов.

        Args:
            texts: Список текстов
            prefer_fasttext: Предпочитать FastText для определения

        Returns:
            Список результатов определения языка для каждого текста
        """
        results = []

        for i, text in enumerate(texts):
            try:
                result = self.detect_language(text, prefer_fasttext)
                results.append(result)
                logger.debug(f"Обработан текст {i+1}/{len(texts)}")
            except Exception as e:
                logger.error(f"Ошибка при определении языка текста {i+1}: {e}")
                results.append(
                    {
                        "language_code": None,
                        "language_name": None,
                        "confidence": 0.0,
                        "method": "error",
                        "is_supported": False,
                        "error": str(e),
                    }
                )

        return results

    def is_language_supported(self, language_code: str) -> bool:
        """
        Проверка поддержки языка.

        Args:
            language_code: Код языка

        Returns:
            True если язык поддерживается
        """
        return language_code in ["en", "ru", "de"]

    def get_language_name(self, language_code: str) -> Optional[str]:
        """
        Получение названия языка по коду.

        Args:
            language_code: Код языка

        Returns:
            Название языка или None
        """
        return self.LANGUAGE_CODES.get(language_code)

    def get_available_methods(self) -> list:
        """
        Получение списка доступных методов определения языка.

        Returns:
            Список доступных методов
        """
        methods = ["langdetect"]
        if self.fasttext_model:
            methods.append("fasttext")
        return methods


# Фабричная функция для удобства
def create_language_detector(
    use_fasttext: bool = False, model_path: Optional[str] = None
) -> LanguageDetector:
    """
    Создание экземпляра LanguageDetector.

    Args:
        use_fasttext: Использовать FastText
        model_path: Путь к модели FastText

    Returns:
        Экземпляр LanguageDetector
    """
    return LanguageDetector(use_fasttext=use_fasttext, model_path=model_path)


# Пример использования
if __name__ == "__main__":
    # Примеры текстов на разных языках
    sample_texts = [
        "This is an example text in English language for testing purposes.",
        "Это пример текста на русском языке для тестирования определения языка.",
        "Dies ist ein Beispieltext in deutscher Sprache zum Testen der Spracherkennung.",
        "Ceci est un texte exemple en français pour tester la détection de langue.",
        "短文本测试",
    ]

    # Создание детектора
    detector = LanguageDetector()

    # Определение языка для каждого текста
    for i, text in enumerate(sample_texts, 1):
        print(f"\nТекст {i}:")
        print(f"Содержимое: {text[:50]}...")

        result = detector.detect_language(text)

        if result["language_code"]:
            print(f"Язык: {result['language_name']} ({result['language_code']})")
            print(f"Уверенность: {result['confidence']:.2f}")
            print(f"Метод: {result['method']}")
            print(f"Поддерживается: {result['is_supported']}")
        else:
            print("Не удалось определить язык")
            if result["error"]:
                print(f"Ошибка: {result['error']}")

    # Пример пакетной обработки
    print("\n\n=== Пакетная обработка ===")
    batch_results = detector.detect_language_batch(sample_texts)

    for i, result in enumerate(batch_results, 1):
        if result["language_code"]:
            print(
                f"Текст {i}: {result['language_name']} "
                f"(уверенность: {result['confidence']:.2f})"
            )
        else:
            print(f"Текст {i}: Не определен")

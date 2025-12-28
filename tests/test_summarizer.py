import unittest
import sys
import os

# Добавляем путь к src для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import create_summarizer


class TestSummarizer(unittest.TestCase):
    """Базовые тесты для суммаризатора."""
    
    def setUp(self):
        """Инициализация суммаризатора перед каждым тестом."""
        self.summarizer = create_summarizer(device='cpu')
    
    def test_english_summarization(self):
        """Тест резюмирования английского текста."""
        text = "This is a test text about machine learning. " \
               "Machine learning is a subset of artificial intelligence. " \
               "It allows computers to learn from data."
        
        result = self.summarizer.summarize(
            text=text,
            language='en',
            compression=30,
            abstractive=True
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.language, 'en')
        self.assertTrue(len(result.summary) > 0)
        self.assertTrue(len(result.summary) < len(text))
    
    def test_language_detection(self):
        """Тест автоматического определения языка."""
        english_text = "This is an English text."
        russian_text = "Это русский текст."
        
        # Английский
        result_en = self.summarizer.detect_language(english_text)
        self.assertEqual(result_en, 'en')
        
        # Русский
        result_ru = self.summarizer.detect_language(russian_text)
        self.assertEqual(result_ru, 'ru')
    
    def test_invalid_text(self):
        """Тест обработки невалидного текста."""
        with self.assertRaises(ValueError):
            self.summarizer.summarize(text="", language='en', compression=30)
        
        with self.assertRaises(ValueError):
            self.summarizer.summarize(text="Short", language='en', compression=30)
    
    def test_compression_levels(self):
        """Тест разных уровней сжатия."""
        text = "This is a longer test text. " * 10
        
        for compression in [20, 30, 50]:
            result = self.summarizer.summarize(
                text=text,
                language='en',
                compression=compression
            )
            self.assertEqual(result.compression, compression)


if __name__ == '__main__':
    unittest.main()
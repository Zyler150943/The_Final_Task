import os
import sys
import tempfile
import unittest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.file_handler import load_text_file, save_text_file


class TestFileHandler(unittest.TestCase):
    """Тесты для работы с файлами."""

    def test_save_and_load_text(self):
        """Тест сохранения и загрузки текстового файла."""
        test_content = "Это тестовый текст\nС новой строкой."

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
            tmp_path = tmp.name

        try:
            # Сохраняем
            success = save_text_file(test_content, tmp_path, overwrite=True)
            self.assertTrue(success)

            # Загружаем
            loaded_content = load_text_file(tmp_path)
            self.assertEqual(loaded_content, test_content)

        finally:
            # Удаляем временный файл
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_encoding_detection(self):
        """Тест определения кодировки."""
        test_content = "Тест UTF-8 текста"

        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(test_content.encode("utf-8"))

        try:
            content = load_text_file(tmp_path)
            self.assertEqual(content, test_content)

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()

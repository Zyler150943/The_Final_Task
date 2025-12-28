#!/bin/bash

# Скрипт для настройки окружения разработки

echo "Setting up Multilingual Summarizer development environment..."

# Создание виртуального окружения
echo "Creating virtual environment..."
python3 -m venv venv

# Активация виртуального окружения
echo "Activating virtual environment..."
source venv/bin/activate

# Установка зависимостей
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Скачивание ресурсов NLTK
echo "Downloading NLTK resources..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Создание необходимых директорий
echo "Creating directories..."
mkdir -p data outputs logs

# Копирование примеров конфигурации
if [ ! -f "config.json" ]; then
    echo "Creating default config file..."
    echo '{"project_name": "Multilingual Summarizer", "version": "1.0.0"}' > config.json
fi

echo "Setup complete! Activate with: source venv/bin/activate"
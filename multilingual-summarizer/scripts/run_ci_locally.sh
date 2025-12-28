#!/bin/bash

# Скрипт для локального запуска CI/CD проверок

set -e

echo "=== Локальный запуск CI/CD проверок ==="

# 1. Проверка форматирования кода
echo "1. Проверка форматирования с black..."
black --check src tests || {
    echo "❌ Black форматирование не прошло. Запустите: black src tests"
    exit 1
}

# 2. Линтинг
echo "2. Проверка линтинга с flake8..."
flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# 3. Проверка импортов
echo "3. Проверка сортировки импортов с isort..."
isort --check-only --diff src tests || {
    echo "❌ Isort проверка не прошла. Запустите: isort src tests"
    exit 1
}

# 4. Запуск тестов
echo "4. Запуск тестов..."
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# 5. Проверка типов
echo "5. Проверка типов с mypy..."
mypy src --ignore-missing-imports --strict

# 6. Статический анализ
echo "6. Статический анализ с pylint..."
pylint src --exit-zero

echo ""
echo "✅ Все проверки пройдены успешно!"
echo "Код готов к коммиту и пуше в репозиторий."
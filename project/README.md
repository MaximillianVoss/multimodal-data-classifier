# Система классификации текстов и изображений

Проект подготовлен для ВКР по теме "Разработка системы классификации изображений и текстовых данных на языке Python с фронтендом на low-code платформе".

## Стек

- `FastAPI` для API и серверной части
- `Gradio` для low-code веб-интерфейса
- `scikit-learn` для обучения моделей
- `SQLite` для хранения истории распознаваний и метаданных моделей
- `pytest` для автоматических тестов

## Структура

- `vkr_classifier` - исходный код приложения
- `tests` - автотесты
- `artifacts` - обученные модели, таблицы и графики для главы о тестировании
- `scripts` - генерация диаграмм, записки и презентации
- `../docs` - записка ВКР, презентация и методические материалы

## Быстрый запуск

```powershell
python -m pip install -r requirements.txt
python main.py
```

После запуска приложение будет доступно по адресу `http://127.0.0.1:8000`.

## Запуск тестов

```powershell
python -m pytest --cov=vkr_classifier --cov-report=term-missing
```

## Генерация моделей и отчетных артефактов

```powershell
python generate_assets.py
python scripts/build_thesis.py
python scripts/build_presentation.py
```

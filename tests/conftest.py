import os

# Модули celery_app и main читают настройки при импорте; реальный ключ в тестах не нужен.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

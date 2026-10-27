# Данные

**Студент:** Семён Викторович Тарлинский

**Датасет:** Amazon Reviews 2018, категория Digital Music 5-core.

**Первоисточник:** [Amazon Review Data 2018](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/index.html). Используется файл [Digital_Music_5.json.gz](https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Digital_Music_5.json.gz).

Файл содержит 169 781 отзыв. `prepare_data.py` скачивает его в `data/raw/`, удаляет повторные пары пользователь–объект, оставляет 5 000 объектов с наибольшим числом положительных взаимодействий в train и делит отзывы по времени на train, validation и test. После очистки остаётся 102 794 взаимодействия, 15 887 пользователей и 5 000 объектов. Подготовленные таблицы создаются в `data/processed/`. В репозиторий данные не загружаются.

## Семинар 1. Lightning.

К коду прилагается [гайд](https://cvr-aug23.pages.deepschool.ru/lectures/lightning/lightning.html) и может быть
удобнее разбирать код параллельно с гайдом.

Решение задачи мультилейбл классификации на примере определения жанра фильма по постеру.


### Датасет

Включает 7254 постеров и 25 жанров.
Скачать датасет и данные можно [отсюда](https://drive.google.com/file/d/1xiNgtQoOmwEZDkmCGvB7EsezU3kgtLpq/view?usp=sharing).

### Подготовка пайплайна

1. Создание и активация окружения
    ```
    python3 -m venv /path/to/new/virtual/environment
    ```
    ```
    source /path/to/new/virtual/environment/bin/activate
    ```

2. Установка пакетов

    В активированном окружении:
    ```
    pip install -r requirements.txt
    ```

3. Настройка ClearML
   - Регистрируемся в [ClearML](https://app.community.clear.ml/), если ещё нет аккаунта.
   - [в своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем "Create new credentials"
   - в консоли пишем `clearml-init` и следуем инструкциям

4. Настраиваем [config.yaml](configs/config.yaml) под себя.
Обратите внимание на `data_config.data_path`, нужно указать папку куда скачали датасет.

### Обучение

Запуск тренировки:

```
PYTHONPATH=. python src/train.py configs/config.yaml
```

### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb).
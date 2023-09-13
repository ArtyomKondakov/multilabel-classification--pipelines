## Задание № 1:
Необходимо решить задачу мультилейбл-классификации
[Ссылка](https://www.kaggle.com/datasets/nikitarom/planets-dataset) на датасет

Этот сорев на kaggle является совместной инициативой Planet и SCCON, демонстрирующей, как спутниковые снимки могут быть использованы для мониторинга вырубки леса и изменения климата со временем. В рамках этой задачи нам необходимо разработать модель multilabel classification, который поможет выявлять как мелкомасштабные, так и крупномасштабные вмешательства в лесах Амазонки.

### Датасет

Количество помеченных изображений в этой задаче ограничено (около 40 000), разбиение на выборки делается рандомно в скрипте.





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

4. Скаченный архив с данными положить в корень проекта и запустить:
    ```
    python src/pars_data.py
    ```

5. Настраиваем [config.yaml](configs/config.yaml) под себя.

### Обучение

Запуск тренировки:

```
PYTHONPATH=. python src/train.py configs/config.yaml
```
### Эксперименты 
https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/f044b780187d4aada30ed5d8d70f583b/output/execution
 
### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb).
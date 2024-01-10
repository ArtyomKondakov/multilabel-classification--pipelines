## Задание № 1:
Задание было выполнено в рамках курса от [deepschool](https://deepschool.ru).
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
    make install_package
    ```

3. Настройка ClearML
   - Регистрируемся в [ClearML](https://app.community.clear.ml/), если ещё нет аккаунта.
   - [в своем профиле ClearML](https://app.community.clear.ml/profile) нажимаем "Create new credentials"
   - в консоли пишем `clearml-init` и следуем инструкциям

4. Скаченный архив с данными положить в корень проекта и запустить:
    ```
    make pars_data
    ```

5. Настраиваем [config.yaml](configs/config.yaml) под себя.

6. Подкючаем DVC
    - необходимо сгенерировать ssh-key командой [ссылка](https://selectel.ru/blog/tutorials/how-to-generate-ssh)
    ```
    ssh-keygen -t rsa
    ```
    - Далее нужно добавить публичную часть ключа на сервер:
    ```
    ssh-copy-id -i /path/to/your/id_rsa.pub <staging_username>@91.206.15.25
    ```
    Теперь можно проверить, что стало пускать без пароля

    - нициализировать dvc в вашем репозитории:

    ```
    #DVC_REMOTE_NAME -- просто то, как вы хотите назвать хост для dvc
    #USERNAME -- username для входа на сервер
    #STAGING_HOST - адрес сервера

    dvc remote add --default $(DVC_REMOTE_NAME) ssh://91.206.15.25/home/a.kondakov/dvc_files
    dvc remote modify $(DVC_REMOTE_NAME) user $(STAGING_USERNAME)
    dvc config cache.type hardlink,symlink
    ```
    Если будет ругаться на то что нет прав на доступ, можно попытаться еще дополнительно указать ему путь до приватного ключа
    ```
    dvc remote modify $(DVC_REMOTE_NAME) keyfile /path/to/your/private_key (пример ~/.ssh/id_rsa)
    ```
    Для того чтобы загрузить модель с серевра нужно сделать 
    ```
    dvc pull --remote DVC_REMOTE_NAME
    ```
    Если вы всё сделали правильно, то должна появится папка model с model.ckpt
    Для преключения между версями делайте git checkout по комитам 'add version_model'
7. Запуск линтеров:
    ```
    make lint
    ```
### Обучение

Запуск тренировки:

```
make train
```
### Эксперименты 

|  | f1 | precision | recall | link to exp |
| --- | --- | --- | --- | --- |
| resnet18 | 0.59 | 0.68 | 0.56 | [Ссылка](https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/f044b780187d4aada30ed5d8d70f583b/output/execution) |
| resnet34 | 0.61 | 0.68 | 0.57 | [Ссылка](https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/22ea7d71b0b645bdbe6e2121a5a6966c/output/execution) |
| resnet50 | 0.61 | 0.68 | 0.58 | [Ссылка](https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/559be76ef2dc4ce3a02f148a6f54ca64/output/execution) |
| efficientnet_b0 | 0.62 | 0.70 | 0.59 | [Ссылка](https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/e57eceb101e64013a711784f294ce246/output/execution) |
| efficientnet_b1 | 0.65 | 0.73 | 0.62 | [Ссылка](https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/d13ba66d196342079fbf05e1c6bef89c/output/execution) |
| efficientnet_b2 | 0.63 | 0.73 | 0.60 | [Ссылка](https://app.clear.ml/projects/7970d8b03a9144b5a1baa2331d3f79c1/experiments/630b5843a30e49e880b9b12a6d5f94f6/output/execution) |
### Инференс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb).

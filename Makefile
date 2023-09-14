DVC_REMOTE_NAME := kondakov

dvc_pull:
	dvc pull --remote $(DVC_REMOTE_NAME)


install_package:
	pip install -r requirements.txt


lint:
	flake8


pars_data:
	python src/pars_data.py


train:
	PYTHONPATH=. python src/train.py configs/config.yaml

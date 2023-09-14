DVC_REMOTE_NAME := kondakov

dvc_pull:
	dvc pull --remote $(DVC_REMOTE_NAME)
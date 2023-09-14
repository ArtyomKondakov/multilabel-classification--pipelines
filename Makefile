DVC_REMOTE_NAME := kondakov_hw_1
STAGING_HOST := 91.206.15.25
STAGING_USERNAME := a.kondakov

dvc_add:
	dvc remote add --default $(DVC_REMOTE_NAME) ssh://$(STAGING_HOST)/home/$(STAGING_USERNAME)/dvc_files
	dvc remote modify $(DVC_REMOTE_NAME) user $(STAGING_USERNAME)
	dvc config cache.type hardlink,symlink
	dvc remote modify $(DVC_REMOTE_NAME) keyfile  ./.ssh/id_rsa.pub
	dvc pull --remote $(DVC_REMOTE_NAME)
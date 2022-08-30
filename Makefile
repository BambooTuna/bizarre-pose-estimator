.PHONY: setup
setup:
	rm -rf ./images ./outputs
	mkdir -p ./images ./outputs
	gsutil -m cp gs://atel-image-source-prd/images/anime/$(NAME).tar.gz /tmp/images.tar.gz
	cd ./images && tar -zxvf /tmp/images.tar.gz
	rm -rf /tmp/images.tar.gz


.PHONY: upload
upload:
	cd ./outputs && tar zcvf /tmp/outputs.tar.gz .
	gsutil -m cp /tmp/outputs.tar.gz gs://atel-image-source-prd/images/keypoints/$(NAME).tar.gz
	rm -rf /tmp/outputs.tar.gz


.PHONY: run
run:
	nvidia-docker run \
		--rm -it --runtime=nvidia \
        -v $PWD:$PWD \
        -w $PWD \
        bizarre-pose-estimator \
        bash -c "python3 -m _scripts.pose_estimator_multi ./images ./outputs ./_train/character_pose_estim/runs/feat_concat+data.ckpt"


.PHONY: step
step:
	setup NAME="$(NAME)"
	run
	upload NAME="$(NAME)"


.PHONY: build
build:
	docker build -t bizarre-pose-estimator _env
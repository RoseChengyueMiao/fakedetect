DOCKER_REPOSITORY := ml-system# For local usage


ABSOLUTE_PATH := $(shell pwd)
IMAGE_VERSION := 0.0.1
WEB_SINGLE_PATTERN := web_single_pattern

WEB_SINGLE_PATTERN_PORT := 8000

.PHONY: build
build:
	sudo docker build -f './Dockerfile' -t ml-system:web_single_pattern_0.0.1 .
		

.PHONY: run
run:
	sudo docker run \
		-d \
		--name $(WEB_SINGLE_PATTERN) \
		-p 8000:8000 \
		$(DOCKER_REPOSITORY):$(WEB_SINGLE_PATTERN)_$(IMAGE_VERSION)

.PHONY: stop
stop:
	sudo docker rm -f $(WEB_SINGLE_PATTERN)

.PHONY: push
push:
	docker push $(DOCKER_REPOSITORY):$(WEB_SINGLE_PATTERN)_$(IMAGE_VERSION)

.PHONY: build_all
build_all: build

.PHONY: run_all
run_all: run

.PHONY: push_all
push_all: push


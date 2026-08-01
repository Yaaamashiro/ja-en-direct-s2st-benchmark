.DEFAULT_GOAL := test

.PHONY: test test-gpu docker-build docker-smoke
test:
	python -m pytest -q -m "not gpu"

test-gpu:
	python -m pytest -q -m gpu

docker-build:
	docker compose build common fairseq
	docker compose build cascade evaluation

docker-smoke:
	docker compose run --rm common corpus validate --profile smoke --dry-run
	docker compose run --rm fairseq s2ut extract-units --profile smoke --dry-run
	docker compose run --rm cascade cascade run --profile smoke --split test --dry-run

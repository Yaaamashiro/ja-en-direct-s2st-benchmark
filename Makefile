.DEFAULT_GOAL := test

.PHONY: test test-gpu
test:
	python -m pytest -q -m "not gpu"

test-gpu:
	python -m pytest -q -m gpu

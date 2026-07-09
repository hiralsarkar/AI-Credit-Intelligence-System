.PHONY: install test lint demo all
install:
	pip install -r requirements.txt
lint:
	ruff check .
test:
	pytest
demo:
	python 04_decision_engine/demo.py
all: lint test

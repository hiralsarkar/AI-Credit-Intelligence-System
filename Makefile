.PHONY: install test lint demo all
install:
	pip install -r requirements.txt
lint:
	ruff check .
test:
	pytest
demo:
	python 04_decision_engine/demo.py
notebooks:
	python tools/run_notebooks.py
all: lint test

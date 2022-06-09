.PHONY: help install lint mypy test docs clean

.DEFAULT: help
help:
	@echo "Available targets: install, lint, mypy, test, docs, clean, format"

install:
	poetry install

lint:
	poetry run pylint probably ${ARGS}

mypy:
	poetry run mypy probably ${ARGS}

test:
	poetry run pytest --doctest-modules --cov=probably --cov-report html --cov-report term --junitxml=testreport.xml tests/ probably/ ${ARGS}

docs:
	poetry run bash -c "cd docs && make html"

clean:
	rm -rf docs/build

format:
	poetry run yapf -r -i probably/ tests/
	poetry run isort probably/ tests/

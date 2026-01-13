.PHONY: help install test lint format clean run

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package and all development dependencies
	python -m pip install -e .
	python -m pip install -r requirements.txt

run:  ## Run the sales agent application
	python sales_agent/main.py

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=sales_agent --cov-report=html --cov-report=term

lint:  ## Run linting checks
	flake8 sales_agent tests
	mypy sales_agent

format:  ## Format code with black
	black sales_agent tests

format-check:  ## Check if code is formatted correctly
	black --check sales_agent tests

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

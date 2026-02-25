.PHONY: install install-dev lint format test build clean docs-serve docs-build docs-deploy help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install with all development dependencies
	pip install -e ".[dev]"
	pre-commit install

lint: ## Run linters (ruff + mypy)
	ruff check src/ tests/
	mypy src/versifai/

format: ## Format code with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=versifai --cov-report=term-missing

build: ## Build the package
	python -m build

clean: ## Remove build artifacts
	rm -rf dist/ build/ *.egg-info src/*.egg-info site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docs-serve: ## Serve docs locally with live reload
	mkdocs serve

docs-build: ## Build docs site (strict mode)
	mkdocs build --strict

docs-deploy: ## Deploy docs to GitHub Pages
	mkdocs gh-deploy --force

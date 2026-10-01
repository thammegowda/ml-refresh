.PHONY: help setup test build pdf serve browser-test check clean

PYTHON ?= python3
PORT ?= 1414

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install build-time Python packages (.venv), Node packages, and Chromium
	$(PYTHON) -m venv .venv
	.venv/bin/python -m pip install -r jupyter/requirements.txt
	npm ci
	npx playwright install chromium

test: ## Run Pyodide, notebook, rendering, and chapter tests
	npm test

check: ## Check one or more long-form chapters quickly: make check IDS="attention dpo"
	node book/check.mjs $(IDS)

build: ## Build dist/: pages, figures, JupyterLite, and the single-page edition
	npm run build

pdf: ## Print dist/refresh.pdf from dist/book.html (run make build first)
	npm run pdf

serve: ## Serve dist/ at http://localhost:$(PORT)/ml-refresh/
	PORT=$(PORT) npm run preview

browser-test: ## Run browser tests against the running preview (make serve)
	npm run test:browser

clean: ## Remove generated output
	rm -rf dist test-results playwright-report

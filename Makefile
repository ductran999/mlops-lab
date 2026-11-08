default: help

help: ## Show help for each of the Makefile commands
	@awk 'BEGIN \
		{FS = ":.*##"; printf "Usage: make ${cyan}<command>\n${white}Commands:\n"} \
		/^[a-zA-Z_-]+:.*?##/ \
		{ printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } \
		/^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' \
		$(MAKEFILE_LIST)

healthcheck: ## check infra is ready
	@python ./scripts/check_infra.py

format: ## auto-format code (ruff format + lint --fix)
	@poetry run ruff check --fix .
	@poetry run ruff format .

lint: ## check code without modifying (ruff check + format --check)
	@poetry run ruff check .
	@poetry run ruff format --check .

hooks: ## run all pre-commit hooks on every file
	@poetry run pre-commit run --all-files

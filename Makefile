.PHONY: black style validate test install serve

install:
	uv pip install -e .

black:
	uv run black manul_agents --line-length 120

validate:
	uv run black . --line-length 120
	uv run flake8 src --count --statistics
	uv run flake8 tests --count --statistics
	uv run mypy src --strict --explicit-package-bases
	uv run mypy tests --strict --explicit-package-bases

test:
	uv run pytest -s

serve:
	uv run python -m src.bot --bot-config-path configs/bot_prod.json --providers-config-path configs/providers.json --db-path db.sqlite --characters-path configs/characters.json --yookassa-config-path configs/yookassa.json --localization-config-path configs/localization.json --tools-config-path configs/tools_prod.json

serve-test:
	uv run python -m src.bot --bot-config-path configs/bot_test.json --providers-config-path configs/providers.json --db-path test_db.sqlite --characters-path configs/characters.json --yookassa-config-path configs/yookassa.json --localization-config-path configs/localization.json --tools-config-path configs/tools.json

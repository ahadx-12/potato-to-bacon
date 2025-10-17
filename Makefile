install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	black --check src tests

fmt:
	black src tests
	ruff check --fix src tests

test:
	pytest -q

run:
	python -m uvicorn src.potatobacon.api.app:app --reload --port 8000

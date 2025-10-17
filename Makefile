.PHONY: fmt lint type test build docker-build run

fmt:
	black src tests
	ruff check --fix .

lint:
	ruff check .
	black --check src tests

type:
	mypy src

test:
	pytest -q

build:
	python -m build || true

docker-build:
	docker build -t potatobacon:dev .

run:
	uvicorn potatobacon.api.app:app --host 0.0.0.0 --port 8000

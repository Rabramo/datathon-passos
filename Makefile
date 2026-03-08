install:
	pip install -r requirements.txt

test:
	pytest -q

cov:
	pytest --cov=src --cov-report=term-missing

train:
	python -m src.pipelines.train

api:
	uvicorn src.api.main:app --reload
.PHONY: setup data eda train test lint serve all

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

data:
	python scripts/download_data.py

eda: data
	python scripts/eda_data.py

train: data
	python scripts/train.py

test:
	pytest --cov=src/heart_risk --cov=api --cov-report=term-missing

lint:
	ruff check .

serve:
	uvicorn api.main:app --host 0.0.0.0 --port 8000

all: data eda train test


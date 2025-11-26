.PHONY: help install run clean test lint format

help:
	@echo "Old Photo Enhancement - Available Commands"
	@echo "=========================================="
	@echo "make install    - Install dependencies"
	@echo "make run        - Run Streamlit app"
	@echo "make clean      - Clean cache & logs"
	@echo "make test       - Run tests"
	@echo "make lint       - Check code quality"
	@echo "make format     - Format code"
	@echo "make docker-build - Build Docker image"
	@echo "make docker-run  - Run Docker container"

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

run-dev:
	streamlit run app.py --logger.level=debug

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf logs/*.log

test:
	pytest tests/

lint:
	pylint modules/ utils/ app.py

format:
	black modules/ utils/ app.py

docker-build:
	docker build -t old-photo-enhancement:latest .

docker-run:
	docker-compose up

docker-stop:
	docker-compose down
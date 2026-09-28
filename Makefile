.PHONY: setup reproduce test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

reproduce:
	.venv/bin/python src/run_analysis.py

test:
	.venv/bin/python -m pytest

clean:
	rm -rf .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -f results/*.csv
	rm -f results/*.png
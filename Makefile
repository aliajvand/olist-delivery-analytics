.PHONY: help setup sample pipeline test clean
help:
	@echo "setup     install dependencies"
	@echo "sample    generate synthetic Olist-shaped data for a smoke test"
	@echo "pipeline  rebuild data/processed from data/raw"
	@echo "test      run the test suite"
	@echo "clean     remove generated artifacts"
setup:
	pip install -r requirements.txt
sample:
	python scripts/make_sample_data.py --out data/sample_raw --orders 8000
	python scripts/run_pipeline.py --raw-dir data/sample_raw --out-dir data/sample_processed
pipeline:
	python scripts/run_pipeline.py
test:
	pytest tests -q
clean:
	rm -rf data/processed models reports/figures
	find . -name __pycache__ -type d -exec rm -rf {} +

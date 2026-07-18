.PHONY: install test doctor seed ui run eval clean

install:
	python -m venv .venv && . .venv/bin/activate && pip install --upgrade pip && pip install -e ".[dev]"

test:
	pytest -q

doctor:
	deep-researcher --doctor

seed:
	python scripts/seed_demo.py

ui:
	streamlit run app/streamlit_app.py

# Usage: make run TOPIC="Ozempic" FILTERS="last 30 days"
run:
	deep-researcher "$(TOPIC)" --filters "$(FILTERS)"

# Usage: make eval TOPIC="Ozempic" FILTERS="last 30 days"
eval:
	python scripts/evaluate.py "$(TOPIC)" --filters "$(FILTERS)"

clean:
	rm -rf data/cache/*.json data/reports/* data/*.sqlite .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

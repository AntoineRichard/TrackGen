PYTHON ?= python3
LATEXMK ?= latexmk

.PHONY: all validate tables pdf lint check clean

all: check

validate:
	$(PYTHON) scripts/validate_corpus.py
	$(PYTHON) scripts/validate_agent_runs.py

tables: validate
	$(PYTHON) scripts/render_tables.py

pdf: tables
	$(LATEXMK) -pdf main.tex
	$(PYTHON) scripts/check_tex_log.py build/main.log

lint:
	chktex -q -n 8 -n 13 -n 24 main.tex

check: validate tables pdf lint

clean:
	$(LATEXMK) -C main.tex

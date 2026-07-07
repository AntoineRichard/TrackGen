# Survey Manuscript

This directory contains the venue-neutral LaTeX source for the robot-racing course
generation survey.

## Prerequisites

The build requires latexmk, pdfLaTeX, BibTeX, ChkTeX, and the TeX Live packages listed
in the survey foundation plan. Python 3.10 or newer is required by the corpus
validation and table-generation pipeline.

## Build

From the repository root, explicitly select the project rc file:

    latexmk -r paper/latexmkrc -cd -pdf paper/main.tex

Alternatively, run latexmk from this directory, where it discovers latexmkrc
automatically:

    latexmk -pdf main.tex

The generated PDF is written to paper/build/main.pdf. Remove generated LaTeX files
with:

    latexmk -r paper/latexmkrc -cd -C paper/main.tex

## Validation Workflow

Run both data validators from the repository root before corpus integration or paper
builds:

    python3 -m paper.scripts.validate_corpus
    python3 -m paper.scripts.validate_agent_runs

The equivalent Make target is:

    make -C paper validate

The agent-run validator checks the four independent 35-column discovery CSVs and their
reports. The corpus validator separately enforces the integrated data contract,
including narrowly documented `NR` counts for non-bootstrap agent search-log rows.

Run `make check` from this directory for the complete validation, table-generation,
PDF, log-checking, and lint pipeline.

## Artifact Status

The manuscript currently contains the stable section contracts and no screened-corpus
claims. Bibliography entries are added to references.bib only after source metadata and
evidence have been verified.

# A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool

A French cassation ruling answers each *ground* (a distinct legal complaint against the challenged judgment) with a specific block of *reasons*. This dataset pairs the two, decision by decision, and labels each pair with its outcome, its doctrinal family (following Boré and Boré, *La cassation en matière civile*, Dalloz, 2023), and the statutory articles it cites. This repository is the pipeline that builds it, the classification grid it runs on, the human gold annotations it is validated against, and the classifier baselines reported in the accompanying paper.

## Installation

```
git clone https://github.com/avrilemay/cassation-grammar.git
cd cassation-grammar
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or later, in a virtual environment (a bare `pip install` on a shared or conda-base interpreter installs into that shared environment instead).

## Getting the data

The published dataset (`cassation_pairs_2016_2025_v1.1.jsonl`) is deposited on Zenodo at [PLACEHOLDER: Zenodo record URL / DOI]. If you received this repository as a full archive, the file is already in `data/`. It is not tracked by git (hundreds of MB). Download it if you only want the labelled pairs, without re-running the pipeline.

Running the four steps below reproduces the classification (status, doctrinal family, article layer) exactly, byte for byte, against the internal reference corpus. The published copy additionally goes through a pseudonymisation pass described in the datasheet on the Zenodo record, so a local re-run reproduces the labels, not the published file byte for byte.

## Step-by-step run

Every step reads and writes only the paths listed in `config.yaml` (resolved via `pipeline/_config.py::load_config()`, overridable with the `CASSATION_GRAMMAR_CONFIG` environment variable). No path is hard-coded. Point the paths under `data:` wherever you want to keep the (large, not committed) working files, then run the four scripts in order.

```
python3 pipeline/01_collect.py     # step 1: collection (or skip, see below)
python3 pipeline/02_zone_and_pair.py
python3 pipeline/03_classify.py
python3 pipeline/04_articles.py
```

1. **`01_collect.py`, collection.** Downloads the full Court of Cassation fund from Judilibre via the PISTE API, one file per decision, into two pickles (`data.raw_pickle`, `data.clean_pickle`). Needs a PISTE API key in the environment variable `collect.api_key_env` (default `PISTE_API_KEY`, request one from [piste.gouv.fr](https://piste.gouv.fr/)). Never put the key in `config.yaml` or any committed file.
2. **`02_zone_and_pair.py`, zoning and pairing.** Reads `data.clean_pickle`, extracts the operative part and cassation type of each ruling, and pairs each ground with its block of reasons. Writes `data.pairs_appariement`.
3. **`03_classify.py`, classification.** Reads `data.pairs_appariement`, applies the cascade (`grids/detectors.py`, accepted grounds) and the rejection grid (`grids/grille.json`, rejected grounds), derives the outcome status, and applies the `hors_moyen` filter. Writes `data.pairs_classified`. See `grids/README.md` for how the two rule sets combine.
4. **`04_articles.py`, article-level layer.** Reads `data.pairs_classified` and produces the article-level tables, joining each statutory article back to the doctrinal family, the fine-grained ground, and the rejection codes of the ground that cited it.

If you already have a local Judilibre dump, point `config.yaml::data.clean_pickle` at it and start at step 2. `02_zone_and_pair.py` only requires the columns `id`, `chamber`, `solution`, `decision_date`, `number`, `text`, `zones` (see the docstring of `01_collect.py`).

## Repository structure

```
cassation-grammar/
├── config.yaml              single configuration file: every input/output path, every parameter
├── requirements.txt
├── LICENSE                  MIT, for the code (the dataset has its own licence, see below)
├── pipeline/                the four numbered steps, plus shared modules and small reference files
│   ├── 01_collect.py .. 04_articles.py
│   ├── _config.py           config.yaml loader, used by every step
│   ├── classify_lib.py, hybrid_lib.py, hors_moyen_rule.py, article_extraction.py, ...
│   └── reference/           small reference files (Code civil article list, excluded decisions)
├── grids/                   the two rule sets classification runs on (see grids/README.md)
│   ├── detectors.py         the cascade: doctrinal family of accepted grounds
│   └── grille.json          the grid: rejection codes of rejected grounds
├── gold/                    human gold annotations, guides, and machine/annotator disagreements
├── baselines/               classifier baselines against the gold set (see baselines/README.md)
│   └── predictions/         raw predictions of the published zero-shot run
├── data/                    published dataset and pipeline outputs, gitignored
```

## Citation

A machine-readable version of this reference is in `CITATION.cff`.

```bibtex
@misc{floro2026grammar,
  author = {Floro, Avrile and Dhorasoo, Tamara and Holzenberger, Nils and Le Goff, Thomas and Viard, Tiphaine and Boritchev, Maria},
  title  = {A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool},
  year   = {2026},
  note   = {Under review}
}
```

## Licences

Two different licences apply, to two different things:

- **The dataset** (the published ground-reasons pairs, on Zenodo) is under the **Licence Ouverte 2.0 / Open Licence 2.0** (Etalab), the same licence as the source Judilibre data. The full terms and the citation are on the Zenodo record, next to the data.
- **The code** (`pipeline/`, `grids/`, `baselines/`) is under the MIT licence (see `LICENSE`).

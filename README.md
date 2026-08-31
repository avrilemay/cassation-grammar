# A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool

A ruling of the French Court of Cassation answers each ground of appeal with its own block of reasons. A ground is one legal complaint against the appealed decision. The dataset pairs each ground with the reasons that answer it. It covers the five civil chambers of the Court, from 2016 to 2025. It holds 121,536 pairs, drawn from 86,464 rulings. Each pair carries its outcome. When the ground is accepted, the pair also carries its doctrinal family. When it is rejected, the pair carries its rejection codes. The families follow Boré and Boré, *La cassation en matière civile*, Dalloz, 2023.

This repository holds the code that builds the dataset. It also holds the two sets of classification rules, the human annotations used to check the labels, and the code for the baselines reported in the paper.

The dataset is on Zenodo at https://doi.org/10.5281/zenodo.21932747. It can also be browsed online at https://grammaire-cassation.fr. That site shows how many grounds cite each statutory article. It also shows how those grounds were decided and which doctrinal families they fall into. The paper is under review. Its reference is at the end of this page.

## Installation

```
git clone https://github.com/avrilemay/cassation-grammar.git
cd cassation-grammar
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

We ran the code on Python 3.13. Python 3.10 or later should work. We suggest a virtual environment, so that these packages stay separate from the rest of your Python installation. The baselines need torch and transformers as well. Install them with `pip install -r baselines/requirements.txt`.

## Getting the data

The dataset file is `cassation_pairs_2016_2025_v1.1.jsonl`. It is on the Zenodo record given above, and the datasheet on that record describes its fields. The file is about 320 MB, so it is not tracked by git. If you only want the labelled pairs, download the file and put it in `data/`. You do not need to run the pipeline for that.

Running the pipeline rebuilds the labels of the published file: the outcome, the doctrinal family and the rejection codes. The last step also builds the article-level tables, which the published file does not carry. A local run does not rebuild the published file itself, for two reasons. The corpus is downloaded from Judilibre as it stands on the day of the run, and ours was collected in May 2026. The published copy also went through a pseudonymisation pass, which the datasheet describes.

## Running the pipeline

Each step reads its input and output paths from `config.yaml`. The file is loaded by `pipeline/_config.py`. The environment variable `CASSATION_GRAMMAR_CONFIG` can point to another configuration file. Set the paths under `data:` to wherever you want to keep the working files. Those files are large, and they are not committed. Then run the four scripts in order, from the root of the repository.

```
python3 pipeline/01_collect.py
python3 pipeline/02_zone_and_pair.py
python3 pipeline/03_classify.py
python3 pipeline/04_articles.py
```

1. `01_collect.py` downloads the decisions of the Court of Cassation from Judilibre, the database in which the Court publishes them. It reaches Judilibre through PISTE, a portal of the French administration. It saves one file per decision, then builds two tables, `data.raw_pickle` and `data.clean_pickle`. The script needs a PISTE API key in an environment variable. The setting `collect.api_key_env` gives the name of that variable. The default name is `PISTE_API_KEY`. A key can be requested at https://piste.gouv.fr/. The endpoint set in `config.yaml` is the production one. Access to it is a separate PISTE approval. A sandbox key only serves a small test subset. Keep the key out of `config.yaml` and out of any committed file.
2. `02_zone_and_pair.py` reads `data.clean_pickle`. It keeps the rulings of the five civil chambers, from 2016 to 2025, that end in a cassation or a rejection. It drops a short list of decisions, checked by hand, that do not decide an appeal on points of law. It reads the operative part of each ruling and derives from it whether the cassation is total or partial. It then pairs each ground with its block of reasons. It writes `data.pairs_appariement`.
3. `03_classify.py` reads `data.pairs_appariement`. It runs the two rule sets on the reasons of each pair and derives the status of the pair. An accepted ground takes its doctrinal family from the rules in `grids/detectors.py`. A rejected ground takes its rejection codes from the grid in `grids/grille.json`. The pairs whose reasons do not answer the ground on its merits are set apart as `hors_moyen`. It writes `data.pairs_classified`. The file `grids/README.md` explains how the two rule sets combine.
4. `04_articles.py` reads `data.pairs_classified` and builds the article-level tables. It lists every statutory article cited in a block of reasons. Each reference is joined to the doctrinal family or the rejection codes of the ground that cites it.

If you already have a Judilibre dump as a folder of one JSON file per decision, point `data.raw_decisions_dir` at it and run `python3 pipeline/01_collect.py --skip-download`. The script then only builds the two tables. If you already have the cleaned table as a pandas pickle, point `data.clean_pickle` at it and start at step 2. That step only needs the columns `id`, `chamber`, `solution`, `decision_date`, `number`, `text` and `zones`.

## Repository structure

```
cassation-grammar/
├── config.yaml              input and output paths, parameters
├── requirements.txt
├── CITATION.cff             the citation below, in machine-readable form
├── LICENSE                  MIT licence, for the code
├── pipeline/                the four numbered steps and their shared modules
│   ├── _config.py           loader of config.yaml
│   └── reference/           the Civil Code articles and the excluded decisions
├── grids/                   the two rule sets, described in grids/README.md
│   ├── detectors.py         the rules for accepted grounds
│   └── grille.json          the grid for rejected grounds
├── gold/                    the human annotations, described in gold/README.md
├── baselines/               the baselines, described in baselines/README.md
│   └── predictions/         predictions of the zero-shot run
└── data/                    dataset and pipeline outputs, not tracked by git
```

## Citation

```bibtex
@misc{floro2026grammar,
  author = {Floro, Avrile and Dhorasoo, Tamara and Holzenberger, Nils and Le Goff, Thomas and Viard, Tiphaine and Boritchev, Maria},
  title  = {A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool},
  year   = {2026},
  note   = {Under review}
}
```

## Licences

The dataset on Zenodo is under the Open Licence 2.0, also called the Licence Ouverte 2.0. That is the open data licence of the French state, and the licence of the source Judilibre data. The full terms are on the Zenodo record, next to the data.

The code in this repository is under the MIT licence. See `LICENSE`.

# A Grammar of French Civil Cassation: A Ground-Level Dataset and Tool

A ruling of the French Court of Cassation answers each ground of appeal with its own block of reasons. A ground is one legal complaint against the appealed decision. The dataset pairs each ground with the reasons that answer it, for the five civil chambers, from 2016 to 2025. It holds 121,536 pairs, drawn from 86,464 rulings. Each pair carries its outcome. When the ground is accepted, the pair also carries its doctrinal family. When it is rejected, the pair carries its rejection codes. The families follow Boré and Boré, *La cassation en matière civile*, Dalloz, 2023.

This repository holds the code that builds the dataset. It also holds the two sets of classification rules, the human annotations used to check the labels, and the code for the baselines reported in the paper.

The dataset is on Zenodo at https://doi.org/10.5281/zenodo.21932747. It can also be browsed online at https://grammaire-cassation.fr. The paper is under review. Its reference is at the end of this page.

## Getting the data

The dataset file is `cassation_pairs_2016_2025_v1.1.jsonl`, about 320 MB. It is on the Zenodo record given above, and the datasheet on that record describes its fields. Download the file and put it in `data/`. Nothing needs to be run for that.

## Running the pipeline

The pipeline rebuilds the labels of the dataset from the Judilibre corpus. Run it only if you want to check or change the classification.

```
pip install -r requirements.txt
python3 pipeline/01_collect.py
python3 pipeline/02_zone_and_pair.py
python3 pipeline/03_classify.py
python3 pipeline/04_articles.py
```

Every path and setting is in `config.yaml`, with comments. Each script explains itself in its docstring. The first step needs a PISTE API key in the environment variable `PISTE_API_KEY`. Keep the key out of any committed file. The baselines have their own requirements file and their own README, in `baselines/`.

## Repository structure

```
cassation-grammar/
├── config.yaml              input and output paths, parameters
├── requirements.txt
├── CITATION.cff             the citation below, in machine-readable form
├── LICENSE                  MIT licence, for the code
├── pipeline/                the four numbered steps and their shared modules
├── grids/                   the two rule sets, described in grids/README.md
├── gold/                    the human annotations, described in gold/README.md
├── baselines/               the baselines, described in baselines/README.md
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

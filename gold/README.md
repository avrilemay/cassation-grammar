# Gold annotations

This directory ships the human annotations used to validate the rule-based classifier (`pipeline/03_classify.py`): two legally trained readers, blind to the machine label, whose few disagreements were adjudicated. It ships no decision text, only identifiers and labels: join on `arret_id` + `moyen_idx` (or the combined `pid`) against the published dataset to recover `moyen_text` and `motivation_text`.

## What is in this directory

| File | Rows | Content |
|---|---:|---|
| `gold_accepte.jsonl` | 200 | one accepted ground per line, its adjudicated doctrinal family |
| `gold_rejete.jsonl` | 200 | one rejected ground per line, its adjudicated rejection families, multi-label |
| `desaccords_machine_accepte.json` | 19 | accepted grounds where the machine label and the gold disagree |
| `desaccords_machine_rejete.json` | 17 | rejected grounds where the machine label and the gold disagree |
| `desaccords_annotators_accepte.json` | 7 | accepted grounds where the two annotators disagreed, with the adjudicated label |
| `desaccords_annotators_rejete.json` | 10 | rejected grounds where the two annotators disagreed, with the adjudicated label |

Two annotation guides sit in `guides/` (`guide_familles_ACCEPTE.html`, `guide_familles_REJETE.html`), in French, the language they were written and used in: doctrinal definitions and short anonymised examples only, no `moyen_text` or `motivation_text` embedded.

## Sample design

400 grounds were drawn from the classified corpus with a fixed random seed (`20260727`), in two independent populations.

- **Accepted grounds**: 50 drawn uniformly at random from the 32,613 accepted grounds (an unbiased estimate of the natural distribution), then 150 drawn stratified by doctrinal family, quotas allocated as evenly as family sizes allow (`VIOLATION`, `MBL`, `VICE_MOTIFS`, `EXCES_OFFICE`, `VICE_FORME`, `DENATURATION` at 24 to 25 each, plus the corpus's single `OMISSION_ULTRA_PETITA` case). Stratified draws exclude anything already drawn, so no duplicate pid across the 200. Of these 200, 3 carry a gold label outside the six quota families: one ground drawn to fill a quota is annotated `autre_indetermine`, one ground drawn for the `EXCES_OFFICE` quota was adjudicated `OMISSION_ULTRA_PETITA`, and the corpus's single machine-labelled `OMISSION_ULTRA_PETITA` case keeps that label. Two of the three appear among the 19 machine disagreements below.
- **Rejected grounds**: 50 drawn uniformly from the 84,460 rejected grounds, then 50 for each of the three Boré rejection families (`IRREC`, `FOND`, `RNSM`), drawn rarest family first so a ground already claimed is not redrawn for a more common one. A ground can carry more than one family label. It counts toward the stratum it was drawn for.

Each annotator used these guides to annotate all 400 grounds, blind to the machine label and to the other annotator's answers. `strate` records which stratum a ground was drawn for (`uniforme` for the 50 natural draws, otherwise the family name), so a reuser can re-weight the stratified draws back to their true corpus share instead of reading the 200-row sample as representative on its own.

## Format

One JSON object per line. `gold_accepte.jsonl`:

```json
{"pid": "5fca25841ea2172a3d0bbd28|0", "arret_id": "5fca25841ea2172a3d0bbd28", "moyen_idx": 0, "pourvoi": "19-21.036", "chambre": "civ3", "annee": 2020, "famille": "VICE_MOTIFS", "seed": 20260727, "strate": "VICE_MOTIFS"}
```

`gold_rejete.jsonl` (multi-label):

```json
{"pid": "5fca25318136b321d6b7e882|0", "arret_id": "5fca25318136b321d6b7e882", "moyen_idx": 0, "pourvoi": "18-24.468", "chambre": "soc", "annee": 2020, "familles": ["RNSM", "IRREC"], "indetermine": false, "seed": 20260727, "strate": "IRREC"}
```

Fields:

- `pid`, `arret_id`, `moyen_idx`: join keys against the published dataset (`pid` is `arret_id|moyen_idx`).
- `pourvoi`, `chambre`, `annee`: source metadata, copied from the annotation record.
- `famille` (accepted grounds): the adjudicated doctrinal family, or `autre_indetermine` / `OMISSION_ULTRA_PETITA` for the 3 grounds outside the six families (see Sample design).
- `familles` + `indetermine` (rejected grounds, multi-label): the adjudicated rejection families, and a boolean for cases the annotators could not assign to any of the three. `indetermine` is `false` on all 200 rows of this campaign.
- `seed`: the sampling seed, `20260727`, repeated on every row for convenience.
- `strate`: the stratum the ground was drawn for (see Sample design above).

Fields present in the internal exports were dropped: the free-text annotation and adjudication notes (kept internal by policy) and the annotation timestamps (no public value).

## Disagreements

All four files carry only identifiers and labels, like the two gold files.

`desaccords_machine_accepte.json` and `desaccords_machine_rejete.json` list every ground where the adjudicated gold differs from the classifier's own output at the time of the campaign (`machine` vs `gold`), with its stratum. 19 of the 200 accepted grounds and 17 of the 200 rejected grounds are listed. The agreement figures reported for the rule-based classifier are computed directly over the gold files, as `baselines/build_training_data.py` prints them: 91.4 % on accepted grounds (180/197, after excluding the 3 grounds outside the six-family nomenclature) and 91.5 % on rejected grounds (183/200, exact multi-label match). They describe this sample, which over-represents rare families by design. Use `strate` to reweight when a corpus-level estimate is needed.

`desaccords_annotators_accepte.json` and `desaccords_annotators_rejete.json` list the 7 accepted and 10 rejected grounds where the two annotators disagreed before adjudication, with each annotator's label (`annotator_a`, `annotator_b`) and the adjudicated label (`final`).

## Two annotators and adjudication

The 400 grounds were annotated independently by two legally trained annotators, blind to the machine label. Raw inter-annotator agreement is 96.5 % on accepted grounds (193/200, Cohen's κ = 0.957, 95 % CI 0.922 to 0.987) and 95.0 % on rejected grounds (190/200, exact multi-label match, κ = 0.932, 95 % CI 0.889 to 0.972). The 17 disagreements were adjudicated jointly: 13 resolved to the second annotator's label, 4 to the first, none to a third label. Every label in this directory is the adjudicated one.

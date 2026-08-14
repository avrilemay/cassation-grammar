# Baselines

This directory measures how close three learned models get to the
rule-based classifier, comparing them against the same adjudicated human
annotations (`gold/`) and with the same metric: two fine-tuned French language models
(JuriBERT, pretrained on legal text, and CamemBERTv2, generic) and one
general-purpose LLM prompted with the doctrine and no training at all.

## The task

There are two classifiers, one per side of the corpus. On accepted grounds, the model
reads the Court's block of reasons and picks exactly one of the six doctrinal
families. On rejected grounds, it decides which of the three rejection
families apply, and several can apply at once. Training labels are the rule
engine's own output, with one exception in the published run: on 199
accepted and 397 rejected training pairs, the silver label was replaced by a
human annotation from an earlier internal campaign. Those labels are not
distributed, so a re-run trains on silver labels only and the training files
differ slightly from the published ones. The gold annotations in `gold/` are
never used for training, only for scoring.

## Installation

```
pip install -r baselines/requirements.txt
```

Adds torch and transformers on top of the root requirements.txt.

## Commands

```
# 1. Build the train/dev data (also prints the "rules" agreement row below)
python3 baselines/build_training_data.py

# 2. Train one model on one side (defaults: 4 epochs accepted / 3 rejected, as published)
python3 baselines/train.py --model juribert --side accepte
python3 baselines/train.py --model juribert --side rejete
python3 baselines/train.py --model camembertv2 --side accepte
python3 baselines/train.py --model camembertv2 --side rejete

# 3. Smoke test (CPU, tiny subset, checks the mechanics only)
python3 baselines/train.py --model juribert --side accepte --smoke
```

Full training needs a GPU. The script uses one automatically when available.

## Published results

Agreement with the adjudicated gold annotations (`gold/`). On the accepted
side, 3 of the 200 annotated grounds fall outside the six families and are
excluded, leaving 197.

| Classifier | Accepted (n=197) | Rejected (n=200) |
|---|---:|---:|
| Rule-based classifier (`pipeline/`) | 91.4 % (180/197) | 91.5 % (183/200) |
| JuriBERT (legal pretraining) | 89.3 % (176/197) | 92.0 % (184/200) |
| CamemBERTv2 (generic French) | 78.7 % (155/197) | 93.5 % (187/200) |
| Zero-shot LLM (Qwen3-32B-AWQ) | 87.3 % (172/197) | 88.5 % (177/200) |

Both fine-tuned models nominally edge past the rules on rejected grounds,
not significantly, and stay below the rules on accepted grounds. The
zero-shot model stays below the rules on both sides. No baseline
significantly beats the rules.

JuriBERT errors concentrate on the cases where the adjudicated gold and the
rules disagree: 14 of the 17 disagreement cases on each side, against 7 of
the 180 agreement cases on the accepted side and 2 of the 183 on the
rejected side. Trained on the labels the rules produce, it inherits their
blind spots. The concentration fades with distance from the silver labels:
CamemBERTv2 misses 7 of 17 (accepted) and 9 of 17 (rejected) disagreement
cases, the zero-shot model 3 of 17 and 6 of 17.

The zero-shot run used Qwen3-32B-AWQ served by
vLLM at temperature 0, thinking disabled, with zero parsing errors, and it
answered over the annotators' full response space (nine options, abstention
included), a harder task than the six-way choice of the fine-tuned models.

## Zero-shot LLM

The script sends each ground's block of reasons, inside one of the two
prompts of `PROMPT_zero_shot.md` (one per side), to any OpenAI-compatible
endpoint. The prompt gives the model the same doctrinal definitions the human
annotators worked from, and nothing else: no example, no closing formula. Both
prompts live in that single file. A revision is a new version number.

```
python3 baselines/llm_zero_shot.py --dry-run   # print 3 assembled prompts, no API call
python3 baselines/llm_zero_shot.py --smoke     # fake backend, checks the mechanics
python3 baselines/llm_zero_shot.py             # full run, reads the variables below
```

The raw predictions of the published run are shipped in
`baselines/predictions/` (identifiers and predicted labels only, no decision
text). Re-scoring them against the gold files reproduces the table above
without any API call or GPU:

```
python3 baselines/llm_zero_shot.py --rescore \
    baselines/predictions/predictions_accepte_qwen3-32b-awq.jsonl \
    baselines/predictions/predictions_rejete_qwen3-32b-awq.jsonl
```

| Variable | Meaning |
|---|---|
| `LLM_BASE_URL` | Base URL of an OpenAI-compatible endpoint (the script appends `/chat/completions`). |
| `LLM_API_KEY` | Bearer token, optional for a local server. |
| `LLM_MODEL` | Model name sent in the request. |

A response that is still not valid JSON after one retry counts as a
disagreement rather than being excluded, so a non-compliant model cannot
shrink its own evaluation set.

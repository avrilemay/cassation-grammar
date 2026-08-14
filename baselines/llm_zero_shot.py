#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Zero-shot LLM baseline for the doctrinal-family task (see baselines/README.md).

For every ground (one complaint against the ruling) in the gold evaluation set,
this script builds one of two published prompts, calls a chat model behind an
OpenAI-compatible endpoint at temperature zero, and parses the JSON answer
against a fixed schema, retrying once on a malformed reply. Predictions are
then scored against the gold annotation with the same accuracy metric as the
fine-tuned baseline in train.py.

Usage: python3 llm_zero_shot.py [--dry-run | --smoke | --rescore FILE ...]
(see baselines/README.md and baselines/PROMPT_zero_shot.md)
"""
import argparse
import json
import os
import random
import requests
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from _config import load_config, resolve  # noqa: E402

# FAMILLES6, BORE3 and the scoring logic (prf, score_accepte, score_rejete) are
# duplicated from baselines/train.py rather than imported, so this script needs no
# torch/transformers, only requests and PyYAML. An edit to either copy must be
# mirrored in the other.
FAMILLES6 = ["VIOLATION", "MBL", "VICE_MOTIFS", "EXCES_OFFICE", "DENATURATION", "VICE_FORME"]
BORE3 = ["RNSM", "IRREC", "FOND"]

# Full label space of the accepted-side prompt (9 options): the six families the
# fine-tuned baseline is trained on, three additional ones too rare in the dataset
# for it to have a class of their own, and a residual catch-all label.
FAMILLES9 = FAMILLES6 + ["OMISSION_ULTRA_PETITA", "CONTRARIETE_JUGEMENTS", "autre_indetermine"]

PROMPT_MARKERS = {
    "accepte": ("PROMPT_ACCEPTE_V1_START", "PROMPT_ACCEPTE_V1_END"),
    "rejete": ("PROMPT_REJETE_V1_START", "PROMPT_REJETE_V1_END"),
}


# Prompt loading
def load_prompt_templates(prompt_md_path):
    """Extract the two canonical prompt templates from PROMPT_zero_shot.md.

    Each template lives between an HTML-comment marker, inside a fenced ```text
    block, so this loader stays a plain string slice, not a markdown parser.
    """
    text = Path(prompt_md_path).read_text(encoding="utf-8")
    out = {}
    for side, (start_marker, end_marker) in PROMPT_MARKERS.items():
        start = text.index(start_marker)
        end = text.index(end_marker)
        block = text[start:end]
        m = re.search(r"```text\n(.*?)\n```", block, re.DOTALL)
        if not m:
            raise ValueError(f"no fenced ```text block found for {side} between the markers")
        template = m.group(1)
        if "{{MOTIVATION}}" not in template:
            raise ValueError(f"{side} prompt template has no {{{{MOTIVATION}}}} placeholder")
        out[side] = template
    return out


def assemble_prompt(template, motivation_text):
    return template.replace("{{MOTIVATION}}", motivation_text.strip())


# Gold and dataset loading
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_texts_for_pids(dataset_path, needed_pids):
    """Makes one pass over the published dataset, collecting the court's reasons
    (the motivation text) for a set of pids. Stops early once every pid has
    been found."""
    found = {}
    remaining = set(needed_pids)
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if not remaining:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            pid = f"{d['arret_id']}|{d['moyen_idx']}"
            if pid in remaining:
                found[pid] = (d.get("motivation_text") or "").strip()
                remaining.discard(pid)
    return found, remaining


# JSON response parsing
def strip_code_fence(s):
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*\n(.*?)\n```\s*$", s, re.DOTALL)
    return m.group(1).strip() if m else s


def parse_accepte(raw):
    obj = json.loads(strip_code_fence(raw))
    if not isinstance(obj, dict) or "famille" not in obj:
        raise ValueError("missing key 'famille'")
    fam = obj["famille"]
    if fam not in FAMILLES9:
        raise ValueError(f"'{fam}' is not one of the 9 allowed codes")
    return {"famille": fam}


def parse_rejete(raw):
    obj = json.loads(strip_code_fence(raw))
    if not isinstance(obj, dict) or "familles" not in obj or "indetermine" not in obj:
        raise ValueError("missing key 'familles' or 'indetermine'")
    fams = obj["familles"]
    if not isinstance(fams, list) or any(f not in BORE3 for f in fams):
        raise ValueError(f"'familles' must be a list drawn from {BORE3}")
    indet = obj["indetermine"]
    if not isinstance(indet, bool):
        raise ValueError("'indetermine' must be a JSON boolean")
    return {"familles": sorted(set(fams)), "indetermine": indet}


PARSERS = {"accepte": parse_accepte, "rejete": parse_rejete}

RETRY_INSTRUCTION = (
    "\n\nTa réponse précédente n'était pas un objet JSON valide selon le format "
    "demandé. Renvoie uniquement l'objet JSON, sans aucun texte avant ou après, "
    "en respectant exactement le schéma indiqué."
)


# Backends
def make_real_backend(base_url, api_key, model, temperature, timeout):
    import requests

    url = base_url.rstrip("/") + "/chat/completions"

    def call(messages):
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = {"model": model, "messages": messages, "temperature": temperature}
        extra = os.environ.get("LLM_EXTRA_BODY")
        if extra:
            body.update(json.loads(extra))
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    return call


def make_smoke_backend(seed=20260801):
    """Fake backend: always answers with a syntactically valid JSON response,
    drawn deterministically from the allowed label space. It does not use the
    network, and it is not meant to produce a meaningful accuracy figure."""
    rng = random.Random(seed)

    def call(messages):
        # Detect side from the schema line each canonical prompt ends on.
        if "UN_CODE_PARMI_LES_NEUF" in messages[0]["content"]:
            return json.dumps({"famille": rng.choice(FAMILLES9)}, ensure_ascii=False)
        n = rng.randint(0, 2)
        fams = sorted(rng.sample(BORE3, k=n)) if n else []
        return json.dumps({"familles": fams, "indetermine": not fams}, ensure_ascii=False)

    return call


# One case, with the single retry
def call_one(backend, side, prompt_text, max_retries=1):
    messages = [{"role": "user", "content": prompt_text}]
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw = backend(messages)
        except requests.HTTPError as e:
            # A rejected request (e.g. prompt longer than the server window)
            # is recorded and scored as a disagreement instead of stopping the
            # run, mirroring the irrecoverable-parse policy below.
            body = getattr(e.response, "text", "")[:300]
            return {"parsed": None, "raw": None, "retried": attempt > 0,
                    "parse_error": f"http error: {e} :: {body}"}
        try:
            parsed = PARSERS[side](raw)
            return {"parsed": parsed, "raw": raw, "retried": attempt > 0, "parse_error": None}
        except Exception as e:  # malformed JSON or schema violation
            last_error = str(e)
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": RETRY_INSTRUCTION},
            ]
    return {"parsed": None, "raw": raw, "retried": True, "parse_error": last_error}


# Scoring, with the same metric used in train.py (see the module docstring above)
def prf(n_ok, n_true, n_pred):
    """Precision, recall and F1 for one family. Copied from train.py so this
    script does not need to import torch."""
    prec = n_ok / max(1, n_pred)
    rec = n_ok / max(1, n_true)
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3)}


def score_accepte(rows):
    """Score predicted doctrinal families against the gold annotation.

    rows: dicts with the gold family and the model's predicted family (or
    None). Only grounds whose gold family is one of the six trained families
    are scored, mirroring the accepted-side evaluation in train.py. A missing
    prediction still counts in the denominator, as a miss."""
    scorable = [r for r in rows if r["family_annotated"] in FAMILLES6]
    ok = sum(r["family_pred"] == r["family_annotated"] for r in scorable)
    conf = {f: Counter() for f in FAMILLES6}
    for r in scorable:
        pred = r["family_pred"] if r["family_pred"] in FAMILLES6 else "__other__"
        conf[r["family_annotated"]][pred] += 1
    par_famille = {
        f: {"n": sum(conf[f].values()),
            **prf(conf[f][f], sum(conf[f].values()), sum(conf[g][f] for g in FAMILLES6))}
        for f in FAMILLES6
    }
    return {
        "n": len(scorable),
        "n_total_gold": len(rows),
        "excluded_out_of_nomenclature": len(rows) - len(scorable),
        "agreement_model": [ok, len(scorable)],
        "par_famille": par_famille,
        "macro_f1": round(float(sum(par_famille[f]["f1"] for f in FAMILLES6) / len(FAMILLES6)), 3),
    }


def score_rejete(rows):
    """Score predicted rejection families against the gold annotation.

    rows: dicts with the gold family set, whether the gold annotation marks
    the ground as indeterminate, and the model's predicted family set.
    Grounds the gold annotation leaves indeterminate are excluded from scoring,
    mirroring the rejected-side evaluation in train.py. Agreement means an
    exact match of the family set, not a partial overlap."""
    scorable = [r for r in rows if not r["indetermine"]]
    ok = sum(r["familles_pred"] == r["bore_annotated"] for r in scorable)
    st = {f: Counter() for f in BORE3}
    for r in scorable:
        for f in BORE3:
            st[f]["true"] += f in r["bore_annotated"]
            st[f]["pred"] += f in r["familles_pred"]
            st[f]["ok"] += (f in r["bore_annotated"] and f in r["familles_pred"])
    par_famille = {f: {"n": st[f]["true"], **prf(st[f]["ok"], st[f]["true"], st[f]["pred"])} for f in BORE3}
    return {
        "n": len(scorable),
        "n_total_gold": len(rows),
        "excluded_indetermine": len(rows) - len(scorable),
        "agreement_model": [ok, len(scorable)],
        "par_famille": par_famille,
        "macro_f1": round(float(sum(par_famille[f]["f1"] for f in BORE3) / len(BORE3)), 3),
    }


def rescore_files(paths, gold_acc_path, gold_rej_path):
    """Score prediction files against the gold labels shipped in gold/, joined
    on pid. Labels embedded in the prediction records are ignored. Missing
    predictions count as misses."""
    gold_acc = {g["pid"]: g for g in read_jsonl(gold_acc_path)}
    gold_rej = {g["pid"]: g for g in read_jsonl(gold_rej_path)}
    rows = {"accepte": [], "rejete": []}
    n_parse_errors = {"accepte": 0, "rejete": 0}
    for path in paths:
        for r in read_jsonl(path):
            pid = r["pid"]
            if pid in gold_acc:
                rows["accepte"].append({"family_annotated": gold_acc[pid]["famille"],
                                        "family_pred": r.get("family_pred")})
                n_parse_errors["accepte"] += bool(r.get("parse_error"))
            elif pid in gold_rej:
                rows["rejete"].append({"bore_annotated": sorted(gold_rej[pid].get("familles") or []),
                                       "indetermine": bool(gold_rej[pid].get("indetermine")),
                                       "familles_pred": sorted(r.get("familles_pred") or [])})
                n_parse_errors["rejete"] += bool(r.get("parse_error"))
            else:
                raise SystemExit(f"{path}: pid {pid} is in neither gold file")
    for side, score in [("accepte", score_accepte), ("rejete", score_rejete)]:
        if rows[side]:
            metrics = score(rows[side])
            metrics["n_parse_errors_after_retry"] = n_parse_errors[side]
            print_summary(side, metrics)


# Main
def build_gold_cases(cfg, dataset_path, gold_acc_path, gold_rej_path, subset_n=None):
    gold_acc = read_jsonl(gold_acc_path)
    gold_rej = read_jsonl(gold_rej_path)
    if subset_n is not None:
        gold_acc = gold_acc[:subset_n]
        gold_rej = gold_rej[:subset_n]
    needed = {g["pid"] for g in gold_acc} | {g["pid"] for g in gold_rej}
    texts, missing = load_texts_for_pids(dataset_path, needed)
    if missing:
        raise SystemExit(f"{len(missing)} gold pid(s) not found in {dataset_path}: "
                          f"{sorted(missing)[:5]}...")
    return gold_acc, gold_rej, texts


def run_side(side, gold_rows, texts, template, backend, out_dir, max_retries, quiet=False):
    predictions = []
    for i, g in enumerate(gold_rows):
        text = texts[g["pid"]]
        prompt_text = assemble_prompt(template, text)
        result = call_one(backend, side, prompt_text, max_retries=max_retries)
        rec = {"pid": g["pid"], "side": side, "retried": result["retried"],
               "parse_error": result["parse_error"], "raw_response": result["raw"]}
        if side == "accepte":
            rec["family_annotated"] = g["famille"]
            rec["family_pred"] = (result["parsed"] or {}).get("famille")
        else:
            rec["bore_annotated"] = sorted(g.get("familles") or [])
            rec["indetermine_annotated"] = bool(g.get("indetermine"))
            rec["familles_pred"] = (result["parsed"] or {}).get("familles", [])
            rec["indetermine_pred"] = (result["parsed"] or {}).get("indetermine")
        predictions.append(rec)
        if not quiet and (i + 1) % 25 == 0:
            print(f"  {side}: {i + 1}/{len(gold_rows)}", flush=True)

    out_path = out_dir / f"predictions_{side}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if side == "accepte":
        rows = [{"family_annotated": r["family_annotated"],
                 "family_pred": r["family_pred"]} for r in predictions]
        metrics = score_accepte(rows)
    else:
        rows = [{"bore_annotated": r["bore_annotated"], "indetermine": r["indetermine_annotated"],
                 "familles_pred": r["familles_pred"]} for r in predictions]
        metrics = score_rejete(rows)

    n_parse_errors = sum(1 for r in predictions if r["parse_error"])
    metrics["n_parse_errors_after_retry"] = n_parse_errors
    metrics_path = out_dir / f"metrics_{side}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    return metrics, out_path, metrics_path


def print_summary(side, metrics):
    a, b = metrics["agreement_model"]
    pct = 100 * a / max(1, b)
    print(f"\n=== {side}: agreement with the gold annotation ===")
    print(f"model {a}/{b} = {pct:.1f}%  (n_gold={metrics['n_total_gold']}, "
          f"parse errors after retry: {metrics['n_parse_errors_after_retry']})")
    print("per family:", {f: metrics["par_famille"][f]["f1"] for f in metrics["par_famille"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--side", choices=["accepte", "rejete", "both"], default="both")
    parser.add_argument("--dry-run", action="store_true",
                         help="Print 3 assembled prompts (2 accepted, 1 rejected), no API calls.")
    parser.add_argument("--smoke", action="store_true",
                         help="Fake backend, small subset, checks the mechanics end to end.")
    parser.add_argument("--smoke-n", type=int, default=20,
                         help="Cases per side under --smoke (default: 20).")
    parser.add_argument("--rescore", nargs="+", metavar="PREDICTIONS_JSONL",
                         help="Score existing prediction files (e.g. baselines/predictions/) "
                              "against the gold labels. No API call.")
    parser.add_argument("--out-dir", default=None, help="Override baselines.llm_zero_shot.out_dir")
    args = parser.parse_args()

    cfg_all = load_config()
    cfg = cfg_all["baselines"]
    llm_cfg = cfg["llm_zero_shot"]

    dataset_path = resolve(cfg["dataset"])
    gold_acc_path = resolve(cfg["gold_accepte"])
    gold_rej_path = resolve(cfg["gold_rejete"])
    prompt_md_path = resolve(llm_cfg["prompt_file"])
    out_dir = Path(args.out_dir) if args.out_dir else resolve(llm_cfg["out_dir"])

    if args.rescore:
        rescore_files(args.rescore, gold_acc_path, gold_rej_path)
        return

    templates = load_prompt_templates(prompt_md_path)

    if args.dry_run:
        gold_acc = read_jsonl(gold_acc_path)[:2]
        gold_rej = read_jsonl(gold_rej_path)[:1]
        needed = {g["pid"] for g in gold_acc} | {g["pid"] for g in gold_rej}
        texts, missing = load_texts_for_pids(dataset_path, needed)
        if missing:
            raise SystemExit(f"dry-run: pid(s) not found in dataset: {missing}")
        print(f"loaded templates from {prompt_md_path}")
        for g in gold_acc:
            print(f"\n{'=' * 80}\n[accepte] pid={g['pid']}  gold famille={g['famille']}\n{'=' * 80}")
            print(assemble_prompt(templates["accepte"], texts[g["pid"]]))
        for g in gold_rej:
            print(f"\n{'=' * 80}\n[rejete] pid={g['pid']}  gold familles={g.get('familles')}\n{'=' * 80}")
            print(assemble_prompt(templates["rejete"], texts[g["pid"]]))
        return

    subset_n = args.smoke_n if args.smoke else None
    gold_acc, gold_rej, texts = build_gold_cases(cfg, dataset_path, gold_acc_path,
                                                  gold_rej_path, subset_n=subset_n)

    if args.smoke:
        backend = make_smoke_backend()
        max_retries = 1
        print(f"--smoke: fake backend, {len(gold_acc)} accepted + {len(gold_rej)} rejected cases, "
              f"no network access")
    else:
        # Names come from config.yaml::baselines.llm_zero_shot, values from the
        # environment, so the same script targets a local server or a commercial API.
        base_url = _read_env(llm_cfg["base_url_env"], required=True)
        api_key = _read_env(llm_cfg["api_key_env"], required=False)
        model = _read_env(llm_cfg["model_env"], required=True)
        temperature = float(llm_cfg.get("temperature", 0.0))
        timeout = int(llm_cfg.get("request_timeout", 120))
        max_retries = int(llm_cfg.get("max_retries", 1))
        backend = make_real_backend(base_url, api_key, model, temperature, timeout)
        print(f"real backend: base_url={base_url} model={model} temperature={temperature}")

    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_metrics = {}
    if args.side in ("accepte", "both"):
        m, pred_path, metrics_path = run_side("accepte", gold_acc, texts, templates["accepte"],
                                               backend, out_dir, max_retries)
        all_metrics["accepte"] = m
        print(f"wrote {pred_path} and {metrics_path}")
        print_summary("accepte", m)
    if args.side in ("rejete", "both"):
        m, pred_path, metrics_path = run_side("rejete", gold_rej, texts, templates["rejete"],
                                               backend, out_dir, max_retries)
        all_metrics["rejete"] = m
        print(f"wrote {pred_path} and {metrics_path}")
        print_summary("rejete", m)
    print(f"\ndone in {time.time() - t0:.1f}s")


def _read_env(name, required):
    import os
    val = os.environ.get(name)
    if required and not val:
        raise SystemExit(f"environment variable {name} is not set (see baselines/README.md)")
    return val


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the train and validation data for the doctrinal-family baselines.

Each pair of a ground (a legal complaint against the ruling) and the court's
reasons already carries a label from the rule-based pipeline: a doctrinal
family when the ground was accepted, a group of rejection families when it
was rejected. Pairs from the gold evaluation set are written out separately
with their text instead, so no baseline is ever scored on data it trained
on, and the rejected pairs are capped per family combination so the most
common one does not dwarf the rest.

The published run also replaced the silver label with a human annotation
from an earlier internal campaign on 199 accepted and 397 rejected training
pairs. Those labels are not distributed, so a re-run trains on silver labels
only and the training files differ slightly from the published ones. The
gold evaluation files are identical.

Usage: python3 build_training_data.py [--out-dir DIR]
"""
import argparse
import hashlib
import json
import sys
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from _config import load_config, resolve  # noqa: E402

FAMILLES6 = ["VIOLATION", "MBL", "VICE_MOTIFS", "EXCES_OFFICE", "DENATURATION", "VICE_FORME"]
BORE3 = ["RNSM", "IRREC", "FOND"]
IRREC_CODES = {"R2a", "R2b", "R2c", "R2d", "R2e", "R2g_irrecevable", "R4"}


def bore_of(r_codes):
    """Group the grid's r-codes into the three rejection families: summary
    rejection (RNSM), inadmissibility (IRREC), and rejection on the merits
    (FOND). See the Task section of baselines/README.md."""
    fams = set()
    for c in r_codes or []:
        fams.add("RNSM" if c == "R1" else "IRREC" if c in IRREC_CODES else "FOND")
    return sorted(fams)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_pid(rows):
    return {r["pid"]: r for r in rows}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_dev(rows_dict, rng):
    """Deterministic 90/10 split, sorted by pid before shuffling for reproducibility."""
    rows = sorted(rows_dict.values(), key=lambda r: r["pid"])
    rng.shuffle(rows)
    n_dev = max(1, len(rows) // 10)
    return rows[n_dev:], rows[:n_dev]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None,
                         help="Override baselines.out_dir from config.yaml")
    args = parser.parse_args()

    cfg = load_config()["baselines"]
    dataset_path = resolve(cfg["dataset"])
    gold_acc_path = resolve(cfg["gold_accepte"])
    gold_rej_path = resolve(cfg["gold_rejete"])
    out_dir = Path(args.out_dir) if args.out_dir else resolve(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    min_chars = cfg.get("min_chars", 50)
    cap_sig = cfg.get("cap_per_signature", 8000)
    seed = cfg.get("seed", 20260729)
    rng = random.Random(seed)

    sha = hashlib.sha256()
    with open(dataset_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    dataset_sha256 = sha.hexdigest()
    print(f"dataset sha256: {dataset_sha256}")

    gold_acc_ann = index_by_pid(read_jsonl(gold_acc_path))
    gold_rej_ann = index_by_pid(read_jsonl(gold_rej_path))
    print(f"gold: {len(gold_acc_ann)} accepted + {len(gold_rej_ann)} rejected pids "
          f"excluded from training data")

    rows_acc = {}
    rows_rej_all = []
    gold_eval_acc = []
    gold_eval_rej = []
    n = 0
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            n += 1
            pid = f"{d['arret_id']}|{d['moyen_idx']}"
            motiv = (d.get("motivation_text") or "").strip()

            if pid in gold_acc_ann:
                g = gold_acc_ann[pid]
                gold_eval_acc.append({
                    "pid": pid, "text": motiv,
                    "family_annotated": g["famille"],
                    "family_machine": d.get("famille_axe1"),
                    "scorable": g["famille"] in FAMILLES6,
                    "stratum": g.get("strate"),
                })
                continue
            if pid in gold_rej_ann:
                g = gold_rej_ann[pid]
                gold_eval_rej.append({
                    "pid": pid, "text": motiv,
                    "bore_annotated": sorted(g.get("familles") or []),
                    "bore_machine": bore_of(d.get("r_codes")),
                    "indetermine": bool(g.get("indetermine")),
                    "stratum": g.get("strate"),
                })
                continue

            if len(motiv) < min_chars:
                continue
            statut = d.get("statut")
            if statut == "accepte" and d.get("famille_axe1") in FAMILLES6:
                rows_acc[pid] = {"pid": pid, "text": motiv, "famille": d["famille_axe1"]}
            elif statut == "rejete":
                if d.get("flag_nsam_groupe_preambule"):
                    continue
                fams = bore_of(d.get("r_codes"))
                if fams:
                    rows_rej_all.append({"pid": pid, "text": motiv, "bore": fams})
    print(f"{n} pairs read from {dataset_path.name}")
    assert len(gold_eval_acc) == len(gold_acc_ann), "some gold accepted pids not found in dataset"
    assert len(gold_eval_rej) == len(gold_rej_ann), "some gold rejected pids not found in dataset"

    gold_eval_acc.sort(key=lambda r: r["pid"])
    gold_eval_rej.sort(key=lambda r: r["pid"])
    write_jsonl(out_dir / "gold_eval_accepte.jsonl", gold_eval_acc)
    write_jsonl(out_dir / "gold_eval_rejete.jsonl", gold_eval_rej)

    # Rule-based floor on the same gold, for reference: see baselines/README.md.
    sc = [g for g in gold_eval_acc if g["scorable"]]
    if sc:
        acc_floor = sum(g["family_machine"] == g["family_annotated"] for g in sc)
        print(f"rule-based floor, accepted (n={len(sc)}): {acc_floor}/{len(sc)} = "
              f"{100 * acc_floor / len(sc):.1f}%")
    rj = [g for g in gold_eval_rej if not g["indetermine"]]
    if rj:
        rej_floor = sum(g["bore_machine"] == g["bore_annotated"] for g in rj)
        print(f"rule-based floor, rejected (n={len(rj)}): {rej_floor}/{len(rj)} = "
              f"{100 * rej_floor / len(rj):.1f}%")

    rows_rej_all.sort(key=lambda r: r["pid"])
    rng.shuffle(rows_rej_all)
    by_sig = Counter()
    rows_rej = {}
    for r in rows_rej_all:
        sig = "+".join(r["bore"])
        if by_sig[sig] < cap_sig:
            by_sig[sig] += 1
            rows_rej[r["pid"]] = r

    train_a, dev_a = split_train_dev(rows_acc, rng)
    train_r, dev_r = split_train_dev(rows_rej, rng)
    write_jsonl(out_dir / "train_accepte.jsonl", train_a)
    write_jsonl(out_dir / "dev_accepte.jsonl", dev_a)
    write_jsonl(out_dir / "train_rejete.jsonl", train_r)
    write_jsonl(out_dir / "dev_rejete.jsonl", dev_r)
    print(f"accepted: train {len(train_a)} / dev {len(dev_a)} | families:",
          dict(Counter(r["famille"] for r in train_a)))
    print(f"rejected: train {len(train_r)} / dev {len(dev_r)} | signatures:",
          dict(Counter("+".join(r["bore"]) for r in train_r)))

    manifest = {
        "dataset": str(cfg["dataset"]), "dataset_sha256": dataset_sha256, "n_pairs_read": n,
        "seed": seed, "min_chars": min_chars, "cap_per_signature": cap_sig,
        "gold_pids_excluded": {"accepte": len(gold_acc_ann), "rejete": len(gold_rej_ann)},
        "sizes": {"train_accepte": len(train_a), "dev_accepte": len(dev_a),
                  "train_rejete": len(train_r), "dev_rejete": len(dev_r)},
        "families_train_accepte": dict(Counter(r["famille"] for r in train_a)),
        "signatures_train_rejete": dict(Counter("+".join(r["bore"]) for r in train_r)),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(f"manifest written to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classifies each pair into its final status and doctrinal family.

For every (ground, reasons) pair from step 2, this step decides the outcome:
rejected, accepted, accepted by way of consequence, no match, conflict, or
hors_moyen (the reasons text is not a response on the merits). Accepted
grounds get a fine-grained doctrinal family from the cascade (the
fixed-order regex families in grids/detectors.py). Rejected grounds get
their r-code family from the grid (the named patterns in grids/grille.json).
The classification logic lives in pipeline/classify_lib.py.

Usage: python3 03_classify.py OUT.jsonl [--check BASELINE.jsonl] [--in IN.jsonl]
"""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402
from hors_moyen_rule import is_hors_moyen  # noqa: E402

_cfg = load_config()
IN_APPAR = resolve(_cfg["data"]["pairs_appariement"])
DEFAULT_OUT = resolve(_cfg["data"]["pairs_classified"])


def _load_classify_ns():
    """Load the classification functions from pipeline/classify_lib.py: the
    compiled grid, the hors_moyen check, and the r-code (rejection-formula)
    helpers. classify_lib compiles the grid once at import time, so nothing
    else needs to read the grid file directly.
    """
    import classify_lib
    return vars(classify_lib)


def classify_one(ns, p):
    """Classify one (ground, reasons) pair through the full checklist: the
    accepted-side cascade, the rejected-side grid, the resulting status,
    then the hors_moyen check, which takes precedence."""
    motiv = p.get("motivation_text", "") or ""
    fine, fam = ns["classify_axe1"](motiv)
    r_hits = ns["scan_r_patterns"](motiv)
    r_codes, _ = ns["post_process_r"](r_hits, motiv)
    r_codes = ns["_apply_C5_gating"](r_codes, r_hits, fam)
    r_codes = ns["_apply_C3"](r_codes)
    r_codes = ns["_apply_collision_R1R3"](r_codes, motiv, p)
    statut, flags, r_final, fam_final, r_codes_trace = ns["derive_statut"](p, fine, fam, r_codes)
    out = dict(p)
    out["cas_fine_axe1_cascade"] = fine
    out["famille_axe1_cascade"] = fam
    out["r_codes_cascade"] = sorted(r_codes)
    out["cas_fine_axe1"] = fine if fam_final is not None else (fine if fam is not None and fam_final is None else None)
    out["famille_axe1"] = fam_final
    out["r_codes"] = sorted(r_final)
    out["statut"] = statut
    out["flags_classif"] = flags
    if r_codes_trace:
        out["r_codes_trace"] = sorted(r_codes_trace)
    if fam_final is None:
        out["cas_fine_axe1"] = None
    # hors_moyen takes precedence: a pair whose reasons text is not really a
    # response on the merits is reclassified here, whatever the checklist
    # above decided.
    if is_hors_moyen(motiv, out["statut"]):
        out["statut"] = "hors_moyen"
        out["famille_axe1"] = None
        out["cas_fine_axe1"] = None
        out["r_codes"] = []
    return out


def main():
    OUT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_OUT
    check = None
    if "--check" in sys.argv:
        check = Path(sys.argv[sys.argv.index("--check") + 1])
    in_appar = IN_APPAR
    if "--in" in sys.argv:
        in_appar = Path(sys.argv[sys.argv.index("--in") + 1])
    ns = _load_classify_ns()
    print(f"grid: {len(ns['PATTERNS_COMPILED'])} patterns | flat-order cascade | hors_moyen filter")
    st = Counter()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(in_appar, encoding="utf-8") as f, open(OUT, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip():
                continue
            o = classify_one(ns, json.loads(line))
            st[o["statut"]] += 1
            g.write(json.dumps(o, ensure_ascii=False) + "\n")
    print("statuses:", dict(st))
    print("written:", OUT)

    if check:
        print(f"\n=== non-regression check vs {check.name} ===")
        diffs = Counter(); n = 0
        with open(OUT, encoding="utf-8") as fa, open(check, encoding="utf-8") as fb:
            for la, lb in zip(fa, fb):
                if not la.strip() and not lb.strip():
                    continue
                a = json.loads(la); b = json.loads(lb); n += 1
                for k in set(a) | set(b):
                    if a.get(k) != b.get(k):
                        diffs[k] += 1
        # line-count guard: zip() truncates silently at the shorter file, so a
        # truncated OUT would otherwise report 0 DIFF on the lines it does have.
        n_out = sum(1 for l in open(OUT, encoding="utf-8") if l.strip())
        n_check = sum(1 for l in open(check, encoding="utf-8") if l.strip())
        print(f"pairs: {n:,}  (out={n_out:,}  baseline={n_check:,})")
        if n_out != n_check:
            print(f"MISMATCHED LINE COUNT: out={n_out} baseline={n_check}")
            sys.exit(1)
        if diffs:
            print("DIFFS:", dict(diffs))
            sys.exit(1)
        print(f"0 DIFF: 03_classify.py reproduces {check.name} exactly.")


if __name__ == "__main__":
    main()

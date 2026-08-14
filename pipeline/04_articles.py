#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Builds the article-level view of the corpus.

Reads the classified pairs from step 3. The first stage lists every code
article referenced by a ground, one row per reference. The second stage adds
the doctrinal family and the r-codes (the labels identifying which formula
closed a rejected ground) to that list. The third stage isolates grounds
that cite no usable article, split into a few categories. The fourth stage
writes a frozen overview of decisions, grounds and article coverage for
reporting. Each stage reads only the previous stage's output.

Usage: python3 04_articles.py [--skip-long] [--skip-enrich] [--skip-sans-article] [--skip-overview]
"""
import json
import sys
from pathlib import Path
from collections import Counter

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402
from article_extraction import map_pair  # noqa: E402

_cfg = load_config()
PAIRS_CLASSIFIED = resolve(_cfg["data"]["pairs_classified"])
OUT_LONG_PARQUET = resolve(_cfg["data"]["article_long_parquet"])
OUT_LONG_CSV = resolve(_cfg["data"]["article_long_csv"])
OUT_ENRICHED = resolve(_cfg["data"]["article_long_enriched"])
OUT_SANS_ARTICLE = resolve(_cfg["data"]["moyens_sans_article"])
OUT_SANS_ARTICLE_SUMMARY = resolve(_cfg["data"]["moyens_sans_article_summary"])
OUT_OVERVIEW = resolve(_cfg["data"]["corpus_overview"])
OUT_REAL_MOYENS_PIDS = resolve(_cfg["data"]["real_moyens_pids"])

LONG_COLS = ["arret_id", "moyen_idx", "pourvoi", "chambre", "annee", "code", "version", "num", "alinea",
             "source_kind", "is_visa", "is_procedural", "et_suivants", "statut", "cassation_type", "solution"]


# Stage 1: one row per distinct article a ground relies on
def build_article_long():
    assert PAIRS_CLASSIFIED.exists(), f"not found: {PAIRS_CLASSIFIED}"
    rows = []
    n_pairs = 0
    with open(PAIRS_CLASSIFIED, encoding="utf-8") as f:
        for line in f:
            n_pairs += 1
            rows.extend(map_pair(json.loads(line)))
    df = pd.DataFrame(rows)[LONG_COLS]
    print(f"{n_pairs} pairs -> {len(df):,} article rows")

    print("=== source_kind ===")
    print(df.source_kind.value_counts().to_string())
    usable = df[df.source_kind.isin(["code", "nu_promu"])]
    print(f"\nusable rows (code|nu_promu): {len(usable):,}")
    print(f"distinct pairs covered (>=1 code): {usable.groupby(['arret_id', 'moyen_idx']).ngroups:,} / {n_pairs:,}")
    print(f"distinct decisions: {df.arret_id.nunique():,}")

    OUT_LONG_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_LONG_PARQUET, index=False)
    df.to_csv(OUT_LONG_CSV, index=False)
    print(f"written: {OUT_LONG_PARQUET} ({OUT_LONG_PARQUET.stat().st_size/1e6:.1f} MB)")
    print(f"written: {OUT_LONG_CSV} ({OUT_LONG_CSV.stat().st_size/1e6:.1f} MB)")
    return df


# Stage 2: adds the doctrinal family, fine-grained ground and r-codes to the long table
def build_article_long_enriched():
    assert OUT_LONG_PARQUET.exists() and PAIRS_CLASSIFIED.exists()
    df = pd.read_parquet(OUT_LONG_PARQUET)
    print("long table:", len(df), "rows")

    # map (arret_id, moyen_idx) -> taxonomy (principal value, falls back to the cascade-only field)
    tax = {}
    with open(PAIRS_CLASSIFIED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            k = (r["arret_id"], r["moyen_idx"])
            fam = r.get("famille_axe1") or r.get("famille_axe1_cascade")
            cf = r.get("cas_fine_axe1") or r.get("cas_fine_axe1_cascade")
            rc = r.get("r_codes") or r.get("r_codes_cascade") or []
            tax[k] = (fam, cf, ";".join(rc))
    print("taxonomized pairs:", len(tax))

    keys = list(zip(df["arret_id"], df["moyen_idx"]))
    df["famille"] = [tax.get(k, (None, None, ""))[0] for k in keys]
    df["cas_fine"] = [tax.get(k, (None, None, ""))[1] for k in keys]
    df["r_codes"] = [tax.get(k, (None, None, ""))[2] for k in keys]

    # Coverage QA is anchored on the pair's final status, not the decision's
    # own outcome. Three of these lines are non-zero by design, not by
    # fault, documented here because nothing else in the repository
    # arbitrates the expected ranges.
    #  - family on PVC (accepted by way of consequence) ~0-5%: status
    #    derivation (pipeline/classify_lib.py) clears the doctrinal family on
    #    most PVC routes. Only one of them keeps it. Verified against the
    #    classified corpus: 2/907 PVC pairs carry a family (0.2%).
    #  - family on rejects, and r-codes on accepted grounds, a few percent:
    #    the lookup above falls back to the pre-arbitration cascade field
    #    when the final field is empty (see this function's docstring), so a
    #    pair whose conflict was resolved to rejected (or accepted) can still
    #    surface the raw cascade signal that the arbitration overrode.
    acc = df[df.statut == "accepte"]
    pvc = df[df.statut == "accepte_par_voie_de_consequence"]
    rej = df[df.statut == "rejete"]
    print(f"famille on accepte: {100*acc.famille.notna().mean():.1f}% covered (expected ~100%)")
    print(f"famille on PVC     : {100*pvc.famille.notna().mean():.1f}% (expected ~0-5%, see note above)")
    print(f"famille on rejects : {100*rej.famille.notna().mean():.1f}% (expected a few %, see note above)")
    print(f"r_codes on rejects : {100*(rej.r_codes!='').mean():.1f}% (expected ~98%)")
    print(f"r_codes on accepte : {100*(acc.r_codes!='').mean():.1f}% (expected a few %, see note above)")
    print("\nfamilies (accepte):", acc.famille.value_counts().to_dict())

    df.to_parquet(OUT_ENRICHED, index=False)
    print(f"written: {OUT_ENRICHED} ({OUT_ENRICHED.stat().st_size/1e6:.1f} MB), +famille +cas_fine +r_codes")
    return df


# Stage 3: grounds with no usable article reference
def build_moyens_sans_article():
    assert PAIRS_CLASSIFIED.exists() and OUT_ENRICHED.exists(), f"missing inputs: {PAIRS_CLASSIFIED}, {OUT_ENRICHED}"

    fam = lambda r: r.get("famille_axe1") or r.get("famille_axe1_cascade")
    rcs = lambda r: r.get("r_codes") or r.get("r_codes_cascade") or []
    moyens = {}
    n_lines = n_neg = n_vide = 0
    with open(PAIRS_CLASSIFIED, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            n_lines += 1
            if r.get("moyen_idx", -1) < 0:
                n_neg += 1
                continue
            if not (r.get("moyen_text") or "").strip():
                n_vide += 1
                continue
            pid = f"{r['arret_id']}|{r['moyen_idx']}"
            moyens[pid] = dict(arret_id=r["arret_id"], moyen_idx=int(r["moyen_idx"]),
                               statut=r.get("statut"), famille=fam(r), r_codes=";".join(rcs(r)),
                               annee=r.get("annee"), chambre=r.get("chambre"),
                               motivation_text=r.get("motivation_text") or "",
                               motiv_start=r.get("motiv_start"), motiv_end=r.get("motiv_end"))
    N = len(moyens)
    print(f"lines {n_lines:,} | excluded moyen_idx<0 {n_neg:,} | empty moyen_text {n_vide:,} | universe N {N:,}")

    dfe = pd.read_parquet(OUT_ENRICHED)
    dfe["pid"] = dfe.arret_id + "|" + dfe.moyen_idx.astype(str)
    sk = dfe.groupby("pid").source_kind.agg(lambda s: set(s))
    dfe_pids = set(sk.index)
    usable_pids = set(sk[sk.apply(lambda s: bool(s & {"code", "nu_promu"}))].index)

    def strate(pid):
        if pid in usable_pids:
            return "d_usable"
        if pid not in dfe_pids:
            return "a_aucune_ref"
        return "c_nu_seul" if "nu" in sk[pid] else "b_non_code_seul"

    for pid, m in moyens.items():
        m["strate"] = strate(pid)
    print("strata computed (a/b/c/d).")

    cols = ["pid", "arret_id", "moyen_idx", "strate", "statut", "famille", "r_codes", "annee", "chambre",
            "motivation_text", "motiv_start", "motiv_end"]
    sans = ("a_aucune_ref", "b_non_code_seul", "c_nu_seul")
    sa = pd.DataFrame([{"pid": pid, **{k: m[k] for k in cols if k != "pid"}}
                        for pid, m in moyens.items() if m["strate"] in sans])[cols]
    OUT_SANS_ARTICLE.parent.mkdir(parents=True, exist_ok=True)
    sa.to_parquet(OUT_SANS_ARTICLE, index=False)
    print(f"written {OUT_SANS_ARTICLE}: {len(sa):,} rows ({OUT_SANS_ARTICLE.stat().st_size/1e6:.1f} MB)")
    print(sa.strate.value_counts().to_string())

    s = pd.Series([m["strate"] for m in moyens.values()]).value_counts()
    lab = {"d_usable": "(d) has a used article [control]", "a_aucune_ref": "(a) no article reference",
           "b_non_code_seul": "(b) out-of-code visa (statute/treaty/ECHR/agreement)",
           "c_nu_seul": "(c) bare number not attached to a code"}
    strates = {k: {"n": int(s.get(k, 0)), "pct": round(s.get(k, 0)/N*100, 1)}
               for k in ["d_usable", "a_aucune_ref", "b_non_code_seul", "c_nu_seul"]}
    sans_n = int(N - s.get("d_usable", 0))
    vent_statut = {k: int(v) for k, v in sa.statut.value_counts(dropna=False).items()}
    vent_strate_statut = {st: {k: int(v) for k, v in sa[sa.strate == st].statut.value_counts().items()} for st in sans}
    rej = sa[sa.statut == "rejete"]
    rc = {}
    for x in rej.r_codes:
        for c in (x.split(";") if x else ["(none)"]):
            rc[c] = rc.get(c, 0) + 1
    rcodes_rej = {k: int(v) for k, v in sorted(rc.items(), key=lambda x: -x[1])[:15]}
    acc = sa[sa.statut == "accepte"]
    fam_acc = {k: int(v) for k, v in acc.famille.fillna("(none)").value_counts().head(10).items()}
    summary = {
        "generated_by": "04_articles.py",
        "universe": "genuine grounds (moyen_idx>=0 & non-empty moyen_text)",
        "denominateur_N": N,
        "excluded": {"moyen_idx_neg": n_neg, "moyen_text_vide": n_vide,
                     "note_moyen_text_vide": "98.6% carry an article (NSAM/split blocks) -> not genuine grounds, excluded"},
        "strates": strates, "sans_article_a_b_c": {"n": sans_n, "pct": round(sans_n/N*100, 1)},
        "ventilation_statut": vent_statut, "ventilation_strate_x_statut": vent_strate_statut,
        "rcodes_rejets_sans_article_multilabel": rcodes_rej, "famille_accepte_sans_article": fam_acc,
        "caveats": ["strict denominator (grounds only): the article-level coverage % changes if"
                    " computed over the wider row universe (all pair rows) instead",
                    "a few 'article A.' grounds (truncated citation) fall in stratum (a) for lack of a"
                    " {L,R,D} prefix in the extractor",
                    "the highlighting trigger (family/r_code) is unavailable for these pids (outside the"
                    " app's paired-highlight table) -> arbitration deferred to a later pass"],
    }
    OUT_SANS_ARTICLE_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=== the 4 strata (denominator N =", N, ") ===")
    for k in ["d_usable", "a_aucune_ref", "b_non_code_seul", "c_nu_seul"]:
        print(f"  {lab[k]:52s} {strates[k]['n']:7,d}  {strates[k]['pct']:5.1f}%")
    print(f"  {'-> WITHOUT article (a+b+c)':52s} {sans_n:7,d}  {round(sans_n/N*100,1):5.1f}%   sum==N: {int(s.sum())==N}")
    print("written", OUT_SANS_ARTICLE_SUMMARY)


# Stage 4: frozen overview reconciling decisions, grounds and article coverage
def build_corpus_overview():
    assert PAIRS_CLASSIFIED.exists() and OUT_ENRICHED.exists() and OUT_SANS_ARTICLE.exists()

    rows, n_neg, n_vide = [], 0, 0
    with open(PAIRS_CLASSIFIED, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("moyen_idx", -1) < 0:
                n_neg += 1
                continue
            if not (r.get("moyen_text") or "").strip():
                n_vide += 1
                continue
            rows.append((f"{r['arret_id']}|{r['moyen_idx']}", r["arret_id"], r.get("statut"), r.get("solution"),
                         r.get("cassation_type"), r.get("annee"), r.get("chambre")))
    m = pd.DataFrame(rows, columns=["pid", "arret_id", "statut", "solution", "cassation_type", "annee", "chambre"])
    OUT_REAL_MOYENS_PIDS.parent.mkdir(parents=True, exist_ok=True)
    m[["pid"]].drop_duplicates().to_parquet(OUT_REAL_MOYENS_PIDS, index=False)  # the exact universe of genuine grounds (scopes the app)
    dec = m.drop_duplicates("arret_id").copy()

    # combined decision type (solution x cassation_type): what a lawyer actually
    # wants, without conflating rejection with an undetermined shape
    def _dtype(sol, ct):
        if sol == "cassation":
            return {"totale": "full cassation", "partielle": "partial cassation"}.get(ct, "cassation (undetermined type)")
        if sol == "rejet":
            return "rejection"
        return sol or "other"
    dec["dtype"] = [_dtype(s, c) for s, c in zip(dec.solution, dec.cassation_type)]

    # article coverage: "without" is a frozen table. "with" = real universe minus "without" (reconciles exactly)
    sans = pd.read_parquet(OUT_SANS_ARTICLE)

    ov = {
        "periode": [int(m.annee.min()), int(m.annee.max())],
        "chambres": sorted(m.chambre.dropna().unique().tolist()),
        "decisions": {
            "total": int(dec.arret_id.nunique()),
            "par_type": {k: int(v) for k, v in dec.dtype.value_counts().items()},
            "par_solution": {k: int(v) for k, v in dec.solution.value_counts(dropna=False).items()},
        },
        "moyens": {
            "total": int(len(m)),
            "par_decision_moy": round(len(m) / dec.arret_id.nunique(), 2),
            "par_decision_max": int(m.groupby("arret_id").size().max()),
            "par_statut": {k: int(v) for k, v in m.statut.value_counts(dropna=False).items()},
            "exclus": {"moyen_text_vide": n_vide, "moyen_idx_negatif": n_neg},
        },
        "articles": {   # reconciles on the real universe: with + without == moyens.total
            "avec_article_code": int(len(m) - len(sans)),
            "sans_article": int(len(sans)),
            "strates_sans": {k: int(v) for k, v in sans.strate.value_counts().items()},
        },
        "labels": {
            "cassation_type": {"totale": "full cassation", "partielle": "partial cassation",
                               "ambigu": "rejection / undetermined (ambiguous)"},
            "statut": {"accepte": "accepted (-> cassation)", "rejete": "rejected",
                       "accepte_par_voie_de_consequence": "accepted by way of consequence",
                       "aucun_match": "no match (classifier undecided)", "conflit": "classification conflict",
                       "hors_moyen": "outside the ground's merits"},
            "strates_sans": {"a_aucune_ref": "no article reference",
                             "b_non_code_seul": "out-of-code visa (statute/treaty/ECHR/agreement)",
                             "c_nu_seul": "bare number not attached to a code"},
        },
    }
    OUT_OVERVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_OVERVIEW.write_text(json.dumps(ov, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", OUT_OVERVIEW)
    print(json.dumps({k: ov[k] for k in ["decisions", "moyens", "articles"]}, ensure_ascii=False, indent=1))


def main():
    argv = set(sys.argv[1:])
    if "--skip-long" not in argv:
        build_article_long()
    if "--skip-enrich" not in argv:
        build_article_long_enriched()
    if "--skip-sans-article" not in argv:
        build_moyens_sans_article()
    if "--skip-overview" not in argv:
        build_corpus_overview()


if __name__ == "__main__":
    main()

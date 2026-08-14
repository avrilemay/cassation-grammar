#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only queries over the article-level table, for exploring the data.

The table comes from pipeline/04_articles.py. Nothing is written back. A
single cassation rate per article would mislead: an article cited only in
the visa (the court's opening list of texts) is cited almost automatically
whenever the ground succeeds. The queries therefore split each article's
occurrences in two: citations confined to the visa, and occurrences
actually discussed in the reasoning, with their outcomes.
"""
from pathlib import Path

import pandas as pd

_FAVORABLE = {"accepte", "accepte_par_voie_de_consequence"}
_USABLE = {"code", "nu_promu"}
CAVEAT_MIN = 20   # minimum occurrences discussed in the reasoning before a co-occurrence caveat is raised
FAIBLE_MIN = 20   # minimum n per breakdown cell before it is flagged as low-count

# 2016 reform (order 2016-131) alias table
# An explicit table for the high-frequency obligations, contracts, and
# liability articles of the corpus. Each row is one rule. Some map one old
# number to several new ones (for example 1134 to 1103, 1104, and 1193).
REFORM_RULES = [
    # (label, old numbers, new numbers)
    ("tortious liability (fault)",             ["1382"], ["1240"]),
    ("tortious liability (negligence)",        ["1383"], ["1241"]),
    ("tortious liability (things/others' acts)", ["1384"], ["1242"]),
    ("tortious liability (animals)",           ["1385"], ["1243"]),
    ("tortious liability (building collapse)", ["1386"], ["1244"]),
    ("damages for non-performance",            ["1147"], ["1231-1"]),
    ("formal notice / damages",                ["1146"], ["1231"]),
    ("foreseeable damages",                    ["1150"], ["1231-3"]),
    ("damages, fraud",                         ["1151"], ["1231-4"]),
    ("penalty clause",                         ["1152"], ["1231-5"]),
    ("default interest",                       ["1153"], ["1231-6"]),
    ("default interest (special)",             ["1153-1"], ["1231-7"]),
    ("binding force / good faith / revocation", ["1134"], ["1103","1104","1193"]),  # 1 -> n
    ("implied content of the contract",        ["1135"], ["1194"]),
    ("privity of contract",                    ["1165"], ["1199"]),
    ("burden of proof",                        ["1315"], ["1353"]),
    ("res judicata",                           ["1351"], ["1355"]),
    ("rescission for non-performance",         ["1184"], ["1224","1225","1226","1227","1228","1229","1230"]),  # 1 -> n
    ("conditions of validity",                 ["1108"], ["1128"]),
    ("defects of consent",                     ["1109"], ["1130"]),
]
# index: num -> (full_group:set, label, role)
_REFORM_IX = {}
for label, anc, nouv in REFORM_RULES:
    grp = set(anc) | set(nouv)
    for n in anc:  _REFORM_IX[n] = (grp, label, "old")
    for n in nouv: _REFORM_IX[n] = (grp, label, "new")


def _reform_field(code, num, base_df, merged=False):
    """Builds the reforme field: the pre or post reform counterpart of this
    article number, its count, and a nudge towards merge_reform=True when
    that matters.

    When the counterpart carries markedly more occurrences, more than three
    times as many, suggest_merge is set and a message recommends
    merge_reform=True: querying this number alone only captures a fraction
    of the rule, and that fraction is also temporally biased, since the new
    number mostly carries post-2016 facts and the old one mostly pre-2016
    facts. This fires on the unbalanced pairs such as 1382/1240 and
    1147/1231-1.
    """
    if code != "civil" or num not in _REFORM_IX:
        return {"has_homologue": False}
    grp, label, role = _REFORM_IX[num]
    homo = sorted(grp - {num})
    counts = {h: int((base_df.num == h).sum()) for h in homo}
    own_n = int((base_df.num == num).sum())
    grp_total = own_n + sum(counts.values())
    pct = round(100 * own_n / grp_total, 1) if grp_total else None
    field = {"has_homologue": True, "regle": label, "role": role, "own_n": own_n,
             "homologues_n": counts, "pct_regle_couverte": pct, "suggest_merge": False}
    biggest = max(counts.values(), default=0)
    if not merged and biggest > 3 * own_n:
        bn = max(counts, key=counts.get)
        periode = "post-2016 facts" if role == "new" else "pre-2016 facts"
        field["suggest_merge"] = True
        field["message"] = (f"This number covers only ~{pct}% of the "
                            f"« {label} » rule (counterpart {bn}, n={counts[bn]}) and a "
                            f"temporal subsample ({periode}). Use merge_reform=True "
                            f"for the whole rule.")
    return field


class ArticleOutcomes:
    """Loads article_long.parquet once and exposes the query functions below."""

    def __init__(self, article_long_path):
        self.path = Path(article_long_path)
        self.df = pd.read_parquet(self.path)

    # 1) outcomes_for_article: a two-part profile, never a single rate
    def _corps_stats(self, corps):
        n = len(corps)
        if n == 0:
            return dict(n=0, pct_accepte=None, pct_rejete=None, pct_pvc=None, pct_hors_moyen=None, pct_autres=None)
        acc = corps.statut.isin(_FAVORABLE).mean()          # accepte + pvc (favorable)
        pvc = (corps.statut == "accepte_par_voie_de_consequence").mean()  # sub-component of "accepte"
        rej = (corps.statut == "rejete").mean()
        hm = (corps.statut == "hors_moyen").mean()
        autres = 1 - acc - rej - hm
        r = lambda x: float(round(100*x, 1))
        return dict(n=n, pct_accepte=r(acc), pct_rejete=r(rej), pct_pvc=r(pvc), pct_hors_moyen=r(hm), pct_autres=r(autres))

    def _breakdown(self, corps, col):
        out = {}
        for key, sub in corps.groupby(col):
            s = self._corps_stats(sub)
            s["n_faible"] = s["n"] < FAIBLE_MIN
            out[str(key)] = s
        return out

    def outcomes_for_article(self, code, num, merge_reform=False, include_nu=False,
                              by_chambre=False, by_annee=False):
        """A two-part profile for one article: how often it is only cited in
        the visa (a count, its weight as a ground), and how the grounds that
        actually discuss it in the reasoning turned out (the signal, read
        off statut).

        Caveat for articles split by the 2016 reform: querying a single
        number of a rule split by order 2016-131 (for example 1240 alone)
        does not return the whole rule, only a subsample from one period,
        since the new number mostly carries post-2016 facts and the old one
        mostly pre-2016 facts. Comparing 1240 to 1382 without merging them
        mixes up the date of the facts with the substance of the rule. The
        reforme field flags this case. Pass merge_reform=True to pool the
        whole rule instead.
        """
        pool = self.df if include_nu else self.df[self.df.source_kind.isin(_USABLE)]
        base = pool[pool.code == code]
        nums = {num}
        if merge_reform and code == "civil" and num in _REFORM_IX:
            nums = set(_REFORM_IX[num][0])
        sub = base[base.num.isin(nums)]
        visa  = sub[sub.is_visa]
        corps = sub[~sub.is_visa]
        cs = self._corps_stats(corps)
        prof = {
            "article": {"code": code, "num": num, **({"merged_nums": sorted(nums)} if len(nums) > 1 else {})},
            "n_total": len(sub),
            "is_procedural": bool(sub.is_procedural.any()),
            "volet_visa":  {"n": len(visa)},          # a count: its weight as a ground for cassation
            "volet_corps": cs,                         # occurrences discussed in the reasoning, by outcome
            "n_arrets_distincts": int(sub.arret_id.nunique()),
            "reforme": _reform_field(code, num, base, merged=(len(nums) > 1)),
            "caveat_cooccurrence": len(corps) >= CAVEAT_MIN,
        }
        if by_chambre: prof["by_chambre"] = self._breakdown(corps, "chambre")
        if by_annee:   prof["by_annee"]   = self._breakdown(corps, "annee")
        return prof

    # 2) top_articles: ranking (procedural articles excluded by default)
    def top_articles(self, code=None, by="visa", exclude_procedural=True, min_n=30, head=20):
        pool = self.df[self.df.source_kind.isin(_USABLE)]
        if code is not None:
            pool = pool[pool.code == code]
        if exclude_procedural:
            pool = pool[~pool.is_procedural]
        rows = []
        for (c, num), sub in pool.groupby(["code", "num"]):
            corps = sub[~sub.is_visa]
            rows.append({
                "code": c, "num": num, "n_total": len(sub),
                "n_visa": int(sub.is_visa.sum()), "n_corps": len(corps),
                "pct_accepte_corps": round(100*corps.statut.isin(_FAVORABLE).mean(), 1) if len(corps) else None,
                "pct_rejete_corps":  round(100*(corps.statut == "rejete").mean(), 1) if len(corps) else None,
                "is_procedural": bool(sub.is_procedural.any()),
            })
        t = pd.DataFrame(rows)
        keymap = {"visa": ("n_visa", "n_total"), "n": ("n_total", "n_total"),
                  "corps_accepte": ("pct_accepte_corps", "n_corps"), "corps_rejete": ("pct_rejete_corps", "n_corps")}
        sortcol, ncol = keymap[by]
        t = t[t[ncol] >= min_n]
        return t.sort_values(sortcol, ascending=False).head(head).reset_index(drop=True)


if __name__ == "__main__":
    import pprint
    import sys
    from _config import load_config, resolve

    _cfg = load_config()
    long_path = resolve(_cfg["data"]["article_long_parquet"])
    if not long_path.exists():
        print(f"{long_path} not found: run 04_articles.py first.")
        sys.exit(1)

    A = ArticleOutcomes(long_path)
    pp = pprint.PrettyPrinter(width=100, sort_dicts=False)

    print("=== consistency QA + spot-check on 10 high-frequency articles ===")
    ok = True
    for num in ["1134", "1382", "1147", "1315", "2224", "1351", "1240", "1792", "1153", "271"]:
        p = A.outcomes_for_article("civil", num)
        nv, nc = p["volet_visa"]["n"], p["volet_corps"]["n"]
        add = (nv + nc == p["n_total"])
        cs = p["volet_corps"]
        pcts = (cs["pct_accepte"] or 0) + (cs["pct_rejete"] or 0) + (cs["pct_autres"] or 0)
        sum_ok = abs(pcts - 100) < 0.5 if nc else True
        ok &= add and sum_ok
        print(f"  art {num:6s} n={p['n_total']:5d} visa+corps={'OK' if add else 'KO'} | "
              f"corps n={nc:5d} acc={cs['pct_accepte']} rej={cs['pct_rejete']} pvc-in-acc={cs['pct_pvc']} sum%={'OK' if sum_ok else 'KO'} | "
              f"reforme={p['reforme'].get('regle','-')} caveat={p['caveat_cooccurrence']}")

    a = A.outcomes_for_article("civil", "1240"); b = A.outcomes_for_article("civil", "1382")
    m = A.outcomes_for_article("civil", "1240", merge_reform=True)
    print("\n=== merge_reform QA (tortious liability: 1240 UNION 1382) ===")
    print(f"  n_total: 1240={a['n_total']} + 1382={b['n_total']} = {a['n_total']+b['n_total']} | merged={m['n_total']}  -> {'OK' if a['n_total']+b['n_total']==m['n_total'] else 'KO'}")
    print(f"\nGLOBAL QA: {'OK' if ok else 'KO'}")

    print("\n=== top_articles(by='visa', civil): weight as a ground ===")
    print(A.top_articles("civil", by="visa", head=8).to_string(index=False))
    print("\n=== top_articles(by='corps_rejete', civil, min_n=50): 'defensive' articles ===")
    print(A.top_articles("civil", by="corps_rejete", min_n=50, head=8).to_string(index=False))

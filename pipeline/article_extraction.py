#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Finds the code articles cited in a ground's reasons text.

For every pair, this module extracts each article reference: which code,
which number, whether it is only cited in the visa (the court's opening
list of texts) or discussed in the body of the reasoning, and whether it
concerns the cassation procedure itself rather than the substance. An
explicit citation such as "article 1240 du code civil" names its code. A
bare number can inherit the Civil Code from the nearest earlier mention of
one. No classification happens in this module.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402

_cfg = load_config()
PROMOTION_MIN = _cfg["articles"]["promotion_min_number"]
CC_JSON = resolve(_cfg["articles"]["code_civil_json"])
CC_KEYS = set(json.load(open(CC_JSON, encoding="utf-8")).keys())

# 1) Named codes and attributive sources
CODES = [
 "rural et de la pêche maritime", "rural",
 "relations entre le public et l'administration", "nationalité", "marchés publics",
 "procédures civiles d'exécution", "procédure civile", "procédure pénale", "organisation judiciaire",
 "juridictions financières", "justice administrative", "justice militaire", "justice pénale des mineurs",
 "action sociale et des familles", "famille et aide sociale", "pensions civiles et militaires de retraite",
 "pensions militaires d'invalidité et des victimes de guerre", "santé publique", "sécurité sociale", "travail",
 "commerce", "artisanat", "assurances", "consommation", "construction et de l'habitation", "monétaire et financier",
 "mutualité", "postes et communications électroniques", "propriété intellectuelle", "tourisme", "communes",
 "cinéma et de l'image animée", "défense", "domaine de l'État", "douanes", "éducation", "électoral",
 "étrangers et droit d'asile", "expropriation pour cause d'utilité publique", "propriété des personnes publiques",
 "collectivités territoriales", "général des impôts", "procédures fiscales", "patrimoine", "recherche", "route",
 "sécurité intérieure", "service national", "sport", "urbanisme", "voirie routière", "commande publique",
 "fonction publique", "transports", "aviation civile", "navigation intérieure", "pensions marins français",
 "ports maritimes", "emploi maritime", "environnement", "énergie", "minier", "forestier", "pêche maritime",
 "agricole et pastoral", "déontologie police", "déontologie municipales", "déontologie architectes",
 "marine marchande",
]
_esc = sorted((re.escape(c) for c in CODES), key=len, reverse=True)
_codes_full = rf"code\s+(?:de\s+la|de\s+l'|des|du|d'|de)?\s*(?:{'|'.join(_esc)})"
_codes_isol = r"code\s+(?:civil|p[ée]nal)"
_abbr       = r"C\.?\s*(?:civ|com|consom|trav|assur|p[ée]n|proc\.?\s*civ)\.?"
CODE_RX = re.compile(rf"(?P<code>{_codes_isol}|{_codes_full}|{_abbr})", re.I)

_ABBR_CANON = {"civ":"civil","com":"commerce","consom":"consommation","trav":"travail",
               "assur":"assurances","pen":"pénal","pén":"pénal","procciv":"procédure civile"}
def _canon_code(phrase: str) -> str:
    p = phrase.lower().strip().rstrip(".")
    m = re.match(r"c\.?\s*(civ|com|consom|trav|assur|p[ée]n|proc\.?\s*civ)", p)
    if m:
        key = m.group(1).replace(".", "").replace(" ", "")
        return _ABBR_CANON.get(key, key)
    p = re.sub(r"^code\s+(?:de\s+la|de\s+l'|des|du|d'|de)?\s*", "", p)
    return p.strip()

# Attributive, non-code sources: they break the Civil Code inheritance
# (statute/decree/contract/agreement/...).
ATTR_SRC = re.compile(r"""
    (?: de\s+(?:la\s+|l'|le\s+|son\s+|sa\s+|ses\s+|ce\s+|cet\s+|cette\s+|ces\s+|leur\s+|leurs\s+|ladite\s+|dudit\s+|la\s+m[êe]me\s+|cette\s+m[êe]me\s+|ce\s+m[êe]me\s+)?
         (?:loi|d[ée]cret|ordonnance|arr[êe]t[ée]|contrat|convention\s+collective|convention|statuts|r[èe]glement\s+int[ée]rieur|protocole|charte|pacte|accord|trait[ée]|directive|bail) )
  | (?: loi\s+(?:n[°o]|du|susvis\w*|pr[ée]cit\w*|organique) )
  | (?: r[èe]glement\s*\(?(?:CE|UE|CEE)\)? )
  | (?: des\s+statuts )
""", re.I | re.VERBOSE)
# Version-dating clauses do not break the inheritance: they are not an attributive source.
VERSION_CTX = re.compile(r"(?:r[ée]daction|issue?|ant[ée]rieur|post[ée]rieur|applicable|modifi|abrog|vigueur|cr[éée]+\s+par|version)", re.I)

# 2) Article anchor and utilities (visa, ranges, positional sources)
REF = re.compile(r"""
    \b(?:(?:des?|de\s+l')\s+)?
    art(?:icles?)?\.?\s*
    (?P<version>[LRD])?\.?\s*
    (?P<number>\d{1,4}(?:[-–]\d+|\.\d+)*\s*[°º]?(?:,\s*alin[ée]as?\s*[\dIVer]*(?=\s*,\s*(?:et|ou)\s+(?:[LRD]\.?\s*)?\d))?
        (?:\s*(?:à|au)\s*(?:[LRD]\.?\s*)?\d{1,4}(?:[-–]\d+|\.\d+)*\s*[°º]?)?
        (?:\s*(?:,\s*(?:et|ou)?|\bet\b)\s*(?:[LRD]\.?\s*)?\d{1,4}(?:[-–]\d+|\.\d+)*\s*[°º]?(?:,\s*alin[ée]as?\s*[\dIVer]*(?=\s*,\s*(?:et|ou)\s+(?:[LRD]\.?\s*)?\d))?(?:\s*(?:à|au)\s*(?:[LRD]\.?\s*)?\d{1,4}(?:[-–]\d+|\.\d+)*\s*[°º]?)?)*
        (?:\s+et\s+suivants?)?(?:\s+ancien\w*)?)
    (?P<alinea>\s*,?\s*(?:alin[ée]as?\s*[\dIVer]*|§\s*\d+|\bI{1,3}\b|\bIV\b|\bV\b))?
""", re.I | re.VERBOSE)

_NUMTOK = re.compile(r"\d{1,4}(?:[-–]\d+)*")
def _expand_numbers(group: str):
    g = group.replace("–", "-")
    et_suiv = bool(re.search(r"et\s+suivants?", g, re.I))
    g = re.sub(r"(?<![\d-])\d{1,3}\s*[°º]|alin[ée]as?\s*[\dIVer]*|§\s*\d+", " ", g, flags=re.I)  # strip standalone sub-paragraph numbers + alinéa/§ before number extraction
    nums = set(_NUMTOK.findall(g))
    for m in re.finditer(r"(\d{1,4})\s*(?:à|au)\s*(\d{1,4})", g, re.I):
        a, b = int(m.group(1)), int(m.group(2))
        if 0 < b - a <= 200:
            nums.update(str(x) for x in range(a, b + 1))
    return sorted(nums, key=lambda x: (len(x), x)), et_suiv

# A visa is a block introduced by "Vu" at the start of a line (a paragraph),
# optionally preceded by a paragraph number "N." (structured post-2016 style)
# or "Réponse de la Cour". The block runs to the first ';'/':' on the line,
# else to the end of the line. Anchoring on start-of-line excludes "au vu de
# ..." mid-sentence. Covers both old-style visas "Vu ... ;" and structured,
# numbered visas "N. Vu ... :".
VISA = re.compile(r"(?m)^[ \t]*(?:Réponse de la Cour\s*)?(?:\d{1,3}\.\s*)?Vu\b", re.I)
def _visa_spans(text: str):
    """Spans (start, end) of each visa: from 'Vu' (at line start) to the first ';'/':' on the line, else end of line."""
    out = []
    for m in VISA.finditer(text):
        nl = text.find("\n", m.end())
        if nl == -1:
            nl = len(text)
        seg = text[m.end():nl]
        ends = [seg.find(c) for c in ";:" if seg.find(c) != -1]
        out.append((m.start(), m.end() + (min(ends) if ends else nl - m.end())))
    return out

PROCEDURAL_ARTS = {"procédure civile": {"1014","455","458","620","624","627","700","978","1009-1","1015"}}

def _source_events(text: str):
    """Positioned attributive sources: (pos, kind, code). kind in {civil, othercode, othersrc}."""
    ev = []
    for mm in CODE_RX.finditer(text):
        canon = _canon_code(mm.group("code"))
        ev.append((mm.start(), "civil" if canon == "civil" else "othercode", canon))
    for mm in ATTR_SRC.finditer(text):
        if VERSION_CTX.search(text[max(0, mm.start()-45):mm.start()]):
            continue  # version-dating clause, not an attributive source
        ev.append((mm.start(), "othersrc", None))
    return sorted(ev)

# ENUM_TAIL: recovers a bare article number cited after "et"/"ou"/"," and
# governed by a code that follows, including when a code phrase intervenes
# ("L. 411-3 ... du COJ et 627 du CPC").
ENUM_TAIL = re.compile(
    rf"(?:\bet\b|\bou\b|,)\s*(?:(?P<v>[LRD])\.?\s*)?"
    rf"(?P<n>\d{{1,4}}(?:[-–]\d+)*)\s+du\s+(?P<same>m[êe]me\s+)?(?:{_codes_isol}|{_codes_full}|code)\b",
    re.I,
)
# Guard: a small bare integer (<=15, no L/R/D prefix, no hyphen) preceded by
# "alinéa(s)..." is an alinea number, not an article: do not capture it.
PRE_ALINEA = re.compile(r"alin[ée]as?\s+[^,;.]{0,10}$", re.I)


# 3) Public functions: extract_article_refs and map_pair
def extract_article_refs(text: str) -> List[Dict[str, Any]]:
    """Every article reference found in text (the reasons text), one entry
    per article number: a range such as "1240 à 1242" is split into one
    entry per number.

    A bare number with no code name of its own is promoted to the Civil
    Code only when all of these hold: the last code mentioned before it is
    the Civil Code, the number is a real Civil Code article (CC_KEYS), no
    version letter is attached, and the number is at least PROMOTION_MIN
    (see config.yaml). Below that threshold, bare numbers cited in
    cassation grounds are more often ECHR articles, EU regulation articles,
    decree articles, or contract clauses than genuine low Civil Code
    numbers, and source proximity alone cannot tell them apart. The choice
    favors precision of the attributed code over recall. The threshold only
    gates this promotion: an explicit citation such as "article 5 du code
    civil" is always kept, whatever the number.

    Each entry carries: code, version (L, R, or D), num, alinea, is_visa,
    is_procedural, and source_kind, one of code, nu_promu, source_non_code,
    or nu.
    """
    text = unicodedata.normalize("NFC", text)
    visas = _visa_spans(text)
    ev = _source_events(text)
    refs = list(REF.finditer(text))
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(refs):
        S, E = m.start(), m.end()
        nxt = refs[i+1].start() if i+1 < len(refs) else len(text)
        own = [(pos, kind, canon) for (pos, kind, canon) in ev if E-2 <= pos < min(nxt, E+60)]
        own = own[0] if own else None
        version = m.group("version")
        alinea = (m.group("alinea") or "").strip(" ,") or None
        is_visa = any(s <= S < e for s, e in visas)
        nums, et_suiv = _expand_numbers(m.group("number"))
        for num in nums:
            base = num.split("-")[0]
            if own:
                _, kind, canon = own
                if kind == "othersrc":
                    code, source_kind = None, "source_non_code"
                else:
                    code, source_kind = canon, "code"
            else:
                left = [k for (pos, k, c) in ev if pos < S]
                inh = left[-1] if left else None
                promu = (inh == "civil" and version is None and num in CC_KEYS
                         and base.isdigit() and int(base) >= PROMOTION_MIN)
                code, source_kind = ("civil", "nu_promu") if promu else (None, "nu")
            is_proc = (code in PROCEDURAL_ARTS and num in PROCEDURAL_ARTS[code])
            out.append(dict(code=code, version=version, num=num, alinea=alinea,
                            source_kind=source_kind, is_visa=is_visa, is_procedural=is_proc,
                            et_suivants=et_suiv, span=(S, E)))
    # ENUM_TAIL pass: bare numbers "(et|ou|,) N du [même] code"
    ref_spans = [(rm.start(), rm.end()) for rm in refs]
    for m2 in ENUM_TAIL.finditer(text):
        S2, E2 = m2.start(), m2.end()
        nraw = m2.group("n")
        if (not m2.group("v")) and nraw.isdigit() and int(nraw) <= 15 \
           and "-" not in nraw and "–" not in nraw and PRE_ALINEA.search(text[:S2]):
            continue  # an alinea number, not an article
        if any(a <= S2 < b for a, b in ref_spans):
            continue  # already anchored by REF
        version2 = m2.group("v")
        is_visa2 = any(s <= S2 < e for s, e in visas)
        if m2.group("same"):
            _left = [(p, k, c) for (p, k, c) in ev if p < S2 and k != "othersrc"]
            code_ev = _left[-1] if _left else None
        else:
            _inside = [(p, k, c) for (p, k, c) in ev if S2 < p < E2 and k != "othersrc"]
            code_ev = _inside[0] if _inside else None
        nums2, et2 = _expand_numbers(nraw)
        for num in nums2:
            if code_ev:
                code, source_kind = code_ev[2], "code"
            else:
                code, source_kind = None, "nu"
            is_proc = (code in PROCEDURAL_ARTS and num in PROCEDURAL_ARTS[code])
            out.append(dict(code=code, version=version2, num=num, alinea=None,
                            source_kind=source_kind, is_visa=is_visa2, is_procedural=is_proc,
                            et_suivants=et2, span=(S2, E2)))
    return out


def map_pair(pair: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One entry per distinct article (code, version, num) cited in this pair's reasons, with the pair's outcome fields attached."""
    refs = extract_article_refs(pair.get("motivation_text", ""))
    seen: Dict[Any, Dict[str, Any]] = {}
    for r in refs:
        key = (r["code"], r["version"], r["num"])
        if key not in seen:
            seen[key] = dict(
                arret_id=pair.get("arret_id"), moyen_idx=pair.get("moyen_idx"),
                pourvoi=pair.get("pourvoi"), chambre=pair.get("chambre"), annee=pair.get("annee"),
                code=r["code"], version=r["version"], num=r["num"], alinea=r["alinea"],
                source_kind=r["source_kind"], is_visa=r["is_visa"], is_procedural=r["is_procedural"],
                et_suivants=r["et_suivants"],
                statut=pair.get("statut"), solution=pair.get("solution"),
                cassation_type=pair.get("cassation_type"),
            )
        else:
            seen[key]["is_visa"] |= r["is_visa"]
    return list(seen.values())


# Self-test: regression cases and tricky edge cases
def _self_test():
    tests = [
        "Vu l'article 1101 du code civil, dans sa rédaction antérieure :",
        "Selon l'article L. 511-7 du code monétaire et financier, une société...",
        "en application de l'article 1014, alinéa 1er, du code de procédure civile, il n'y a pas lieu",
        "Vu les articles 1240 à 1242 et 1382 du code civil ;",
        "Vu l'article 6, § 1, de la Convention de sauvegarde des droits de l'homme",
        "Vu l'article 1134 du code civil, dans sa rédaction antérieure, ensemble l'article 1147",   # civil inheritance (>=100)
        "Vu l'article 2262 du code civil, l'article 26, II, de la loi susvisée",                      # 26 not promoted (statute + <100)
        "l'article L. 411-1 du code rural et de la pêche maritime",
    ]
    for t in tests:
        print("TXT:", t)
        for r in extract_article_refs(t):
            print(f"   code={str(r['code']):22} v={r['version']} num={r['num']:7} kind={r['source_kind']:16} "
                  f"visa={r['is_visa']} proc={r['is_procedural']}")
        print()

    def _visa_flags(t):
        return [r["is_visa"] for r in extract_article_refs(t)]

    regressions = [
        ("Vu l'article 1184 du code civil dans sa rédaction antérieure à l'ordonnance n° 2016-131 du 10 février 2016, ensemble l'article L. 1221-1 du code du travail ;",
         [True, True]),                                                   # visa ';', multi-article ("ensemble") -> both
        ("Réponse de la Cour\n\n3. Vu l'article 441, alinéa 2, du code civil :\n\n4. Selon ce texte...",
         [True]),                                                         # structured numbered paragraph, post-2016 style
        ("Vu l'article 1014 du code de procédure civile ;",
         [True]),                                                         # old-style simple ';' visa
        ("La cour d'appel, au vu de l'article 1240 du code civil, a statué : elle a condamné.",
         [False]),                                                        # negative: "au vu de" mid-sentence
        ("Vu l'article 1014 du code de procédure civile ;\n\nSelon l'article 1240 du code civil, ...",
         [True, False]),                                                  # negative: article after the visa, in the reasoning
    ]
    for t, expected in regressions:
        got = _visa_flags(t)
        assert got == expected, f"is_visa REGRESSION: {got} != {expected} | {t[:60]}"
    print("is_visa regression tests: OK")


if __name__ == "__main__":
    _self_test()

    # Validation on a real sample of the classified corpus, if available.
    pairs_path = resolve(load_config()["data"]["pairs_classified"])
    if pairs_path.exists():
        sample_n = 2000
        rows = []
        with open(pairs_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= sample_n:
                    break
                rows.append(json.loads(line))

        all_refs, kinds = [], Counter()
        pairs_with_civil = 0
        for r in rows:
            refs = map_pair(r)
            if any(x["code"] == "civil" and x["source_kind"] in ("code", "nu_promu") for x in refs):
                pairs_with_civil += 1
            all_refs.extend(refs)
            for x in refs:
                kinds[x["source_kind"]] += 1

        print(f"\n{len(rows)} pairs -> {len(all_refs)} distinct article entries")
        print(f"pairs with >=1 Civil Code article (code|nu_promu): {pairs_with_civil} ({100*pairs_with_civil/len(rows):.1f}%)")
        print("source_kind:", dict(kinds))

        promu_low = sum(1 for x in all_refs if x["source_kind"]=="nu_promu" and x["num"].split("-")[0].isdigit() and int(x["num"].split("-")[0])<PROMOTION_MIN)
        expl_low  = sum(1 for x in all_refs if x["source_kind"]=="code" and x["code"]=="civil" and x["num"].split("-")[0].isdigit() and int(x["num"].split("-")[0])<PROMOTION_MIN)
        print(f"\nthreshold check: nu_promu <{PROMOTION_MIN} = {promu_low} (must be 0) | explicit civil articles <{PROMOTION_MIN} preserved = {expl_low}")
    else:
        print(f"\n(skipping the corpus sample validation: {pairs_path} not found)")

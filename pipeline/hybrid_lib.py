"""Assigns each accepted ground its doctrinal family.

A ground is one legal complaint against the appealed decision. When the
Court accepts it, this module reads the Court's block of reasons and
decides which doctrinal family the cassation belongs to (violation of the
law, lack of legal basis, distortion, and so on). The regexes doing the
work live in grids/detectors.py and are applied in a fixed order.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402

_cfg = load_config()
sys.path.insert(0, str(resolve(_cfg["classify"]["detectors_dir"])))
import detectors as D  # noqa: E402

F = re.IGNORECASE
# Markers read by the mode 'B' rerouting guards.
RX = {
 "art4":  re.compile(r"article\s+4\s+(?:du\s+)?(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile", F),
 "art16": re.compile(r"article\s+16\s+(?:du\s+)?(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile", F),
 "art455": re.compile(r"article\s+455\s+(?:du\s+)?(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile", F),
 "art463": re.compile(r"article[s]?\s+463\s+(?:du\s+)?(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile", F),
 "art624": re.compile(r"article\s+624\s+(?:du\s+)?(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile", F),
 "modifie_objet": re.compile(r"modifi[eé]\s+l['’ ]?\s*objet\s+du\s+litige", F),
 "termes_litige": re.compile(r"m[eé]connu\s+les?\s+termes?\s+du\s+litige", F),
 "contradiction": re.compile(r"principe\s+de\s+la\s+contradiction|droits?\s+de\s+la\s+d[eé]fense", F),
 "omis_statuer": re.compile(r"om(?:is|ettant)\s+(?:ainsi\s+)?de\s+statuer", F),
 "voie_conseq": re.compile(r"par\s+voie\s+de\s+cons[eé]quence", F),
 "satisfait_exig": re.compile(r"n['’ ]?a\s+pas\s+satisfait\s+aux\s+exigences", F),
}


# zone_verdict anchors "voie_conseq" to the operative part of the ruling, not
# to a descriptive "Portée et conséquences" block elsewhere in the text.
_RXS_DEBUT_VERDICT = [
    re.compile(p, F) for p in [
        r"qu['’]en\s+statuant\s+ainsi",
        r"\ben\s+statuant\s+ainsi",
        r"qu['’]en\s+se\s+d[eé]terminant\s+ainsi",
        r"\ben\s+se\s+d[eé]terminant\s+ainsi",
        r"\ben\s+se\s+prononçant\s+ainsi",
        r"qu['’]en\s+se\s+prononçant\s+ainsi",
        r"d['’]o[uù]\s+il\s+(?:r[eé]sulte|suit)\s+qu['’]",
        r"\ben\s+proc[eé]dant\s+ainsi",
        r"qu['’]en\s+proc[eé]dant\s+ainsi",
        r"qu['’]?en\s+statuant\s+comme\s+(?:elle|il|ils|elles)\s+(?:l['’]a|l['’]ont|ont|a|avait|avaient)?\s*fait",
        r"qu['’]?en\s+se\s+d[eé]terminant\s+par\s+(?:ces|de\s+tels|tels)\s+motifs",
        r"qu['’]?en\s+(?:r[eé]duisant|condamnant|d[eé]boutant|allouant|refusant|fixant|d[eé]clarant|jugeant|d[eé]cidant|ordonnant|prononçant|admettant|excluant|retenant|rejetant|s['’]abstenant)\s+ainsi",
    ]
]
_RXS_FIN_VERDICT = [
    re.compile(p, F) for p in [
        r"Port[eé]e\s+et\s+cons[eé]quences?\s+de\s+la\s+cassation",
        r"Port[eé]e\s+de\s+la\s+cassation",
        r"Cons[eé]quences?\s+de\s+la\s+cassation",
    ]
]


def zone_verdict(txt: str) -> str:
    """Slice of txt covering the operative part of the ruling."""
    if not isinstance(txt, str) or not txt:
        return txt or ""
    best_start = -1
    for rx in _RXS_DEBUT_VERDICT:
        for m in rx.finditer(txt):
            if m.start() > best_start:
                best_start = m.start()
    if best_start == -1:
        return txt
    zone = txt[best_start:]
    for rx in _RXS_FIN_VERDICT:
        m = rx.search(zone)
        if m:
            zone = zone[:m.start()]
            break
    return zone


def markers(txt: str) -> dict:
    t = txt if isinstance(txt, str) else ""
    out = {k: bool(rx.search(t)) for k, rx in RX.items()}
    # voie_conseq must sit in the operative part of the ruling, not in a
    # descriptive PVC block. If "Portée et conséquences de la
    # cassation" exists, only count "par voie de conséquence" occurrences that
    # precede that block. Otherwise the PVC block is merely descriptive
    # (cassation propagated from another ground) and must not flip the
    # current pair's classification.
    import re as _re
    rx_pvc_block = _re.compile(r"Port[eé]e\s+et\s+cons[eé]quences?\s+de\s+la\s+cassation", _re.IGNORECASE)
    m_pvc = rx_pvc_block.search(t)
    if m_pvc:
        verdict_zone = t[:m_pvc.start()]
        out["voie_conseq"] = bool(RX["voie_conseq"].search(verdict_zone))
    # voie_conseq is required to sit strictly inside the operative part. This
    # check takes precedence over the one above.
    zv = zone_verdict(t)
    if zv != t:
        out["voie_conseq"] = bool(RX["voie_conseq"].search(zv))
    # art463 alone is too broad (topic vs. verdict): require the co-occurrence
    # of an "omis/omettant" formula in the same pair.
    if out["art463"] and not out["omis_statuer"]:
        out["art463"] = False
    out["omission_garde"] = omission_garde(txt)
    return out




# OMISSION_ULTRA_PETITA guard: the "article 463" false friend. When the
# failure-to-rule allegation is itself the subject matter of the proceedings
# (a request for rectification, visas 462/463/464), the actual ground for
# cassation is a violation of that rectification procedure, not omission
# itself. Only the ultra petita branch genuinely belongs to this family.
# Checked against a doctrinal audit (13/13 confirmed). Pitfall: the "463"
# test must be scoped to the visa (the only true positive cites 463 outside
# the visa), and attribution alone is not sufficient.
_OMI_MARQ = re.compile(r"om(?:i[st]|ission)\w*\s+de\s+statuer|ultra\s*.?petita|chose[s]?\s+non\s+demand|accord[eé]\s+plus\s+qu", F)


def _omi_visa(t):
    return " | ".join(m.group(1) for m in re.finditer(r"Vu\s+(?:l['’]article|les\s+articles)(.{0,400}?)[:;]", t or "", re.S))


def _omi_segment_decisif(t):
    ms = list(re.finditer(r"(?:Qu['’])?[Ee]n\s+statuant\s+ainsi", t or ""))
    if not ms:
        return ""
    seg = (t or "")[ms[-1].start():]
    c = re.search(r"\n\s*\n|\n\s*\d+\.\s", seg)
    return seg[:c.start()] if c else seg


def omission_garde(t):
    """True if the OMISSION_ULTRA_PETITA label is doctrinally well-founded."""
    T, V, D_ = t or "", _omi_visa(t), _omi_segment_decisif(t)
    # G0, positive priority path: ultra petita established by confronting the
    # claim against the ruling itself.
    if (re.search(r"seulement sur ce qui est demand|accord[eé] plus qu['’]il n['’]a [eé]t[eé] demand", T)
            and re.search(r"L['’]arr[eê]t\s+(?:condamne|alloue|fixe)|le jugement\s+(?:condamne|alloue)", T)
            and re.search(r"avait (?:demand[eé]|sollicit[eé])|n['’](?:a|avait) pas demand", D_)):
        return True
    # G1, visa of the 462/463/464 regime -> the proceedings are themselves
    # about the rectification request -> VIOLATION, not omission.
    if re.search(r"\b46[234]\b", V) and re.search(r"proc[eé]dure civile", V):
        return False
    # G2, the omission claim is the very object of the reviewed proceedings.
    if re.search(r"requ[eê]te\s+(?:en|au titre de l['’])\s*omission de statuer"
                 r"|(?:r[eé]paration|r[eé]parer|rectification|rectifier|constater)[^.]{0,60}omission[s]?\s+de\s+statuer", T):
        return False
    # G3, refusal to rule -> ultra vires, not omission.
    if re.search(r"(?:lui\s+)?appartenait de statuer|dont (?:elle|il) [eé]tait saisie|refus[eé] de statuer", D_):
        return False
    # G4, marker absent from the decisive segment, or attributed to a
    # different, earlier decision.
    h = re.search(_OMI_MARQ, D_)
    if not h:
        return False
    if re.search(r"pr[eé]c[eé]dent arr[eê]t|arr[eê]t du \d|jugement du \d|d[eé]cision du \d|premiers juges|requ[eê]te", D_):
        return False
    # G5, omission genuinely imputed to the decision under review (narrow branch).
    return True


def _norm(txt: str) -> str:
    return txt.replace("&apos;", "'").lower() if isinstance(txt, str) else ""


def segment_flags(seg_text: str, visa_cite: str = "") -> dict:
    """T_/J_ detectors on the segment text (visa text appended as a safety net)."""
    hay = (seg_text or "")
    if visa_cite:
        hay = hay + "\n" + str(visa_cite)
    norm = _norm(hay)
    return D.detect_axe1(norm, "", hay)


def cascade_segment(flags: dict) -> str | None:
    """The cascade alone on one segment, without the rerouting guards below."""
    return D._cascade_axe1(flags, short=False)


def cascade_segment_rules(flags: dict, mk: dict) -> str | None:
    """Mode 'B': the cascade plus the rerouting guards on VIOLATION and MBL calls."""
    base = D._cascade_axe1(flags, short=False)
    # Demote OMISSION calls that are not doctrinally founded (the "article 463"
    # false friend in the upstream detector).
    if base == "OMISSION_ULTRA_PETITA" and not mk.get("omission_garde", True):
        flags2 = dict(flags)
        flags2["T_omission_ultra_petita"] = False
        base = D._cascade_axe1(flags2, short=False)
    # Reroute MBL_PUR to DEFAUT_REPONSE (family VICE_MOTIFS) when the text couples
    # the "satisfait aux exigences" formula with article 455.
    if base == "MBL_PUR" and mk["satisfait_exig"] and mk["art455"]:
        base = "DEFAUT_REPONSE"
    # Reroute a generic VIOLATION call to the more specific ground whose markers
    # are present in the same text.
    if base in ("VIOL_DIRECTE", "VIOL_MAUVAISE_LECTURE"):
        if mk["modifie_objet"] or mk["art4"]:
            base = "EXCES_POUVOIR"
        elif mk["voie_conseq"]:
            base = "PERTE_FONDEMENT"
        elif (mk["omis_statuer"] or mk["art463"]) and mk.get("omission_garde", False):
            base = "OMISSION_ULTRA_PETITA"  # guarded reroute
    return base


def classify_segment(seg_text: str, visa_cite: str, mode: str) -> tuple[str | None, str | None]:
    """(fine ground, family) for one segment. Mode 'B', the one the pipeline
    uses, applies the cascade and then the checks defined above. Mode 'A'
    applies the cascade alone."""
    flags = segment_flags(seg_text, visa_cite)
    if mode == "B":
        mk = markers((seg_text or "") + "\n" + str(visa_cite or ""))
        fine = cascade_segment_rules(flags, mk)
    else:
        fine = cascade_segment(flags)
    fam = D.FAMILLE_MAPPING.get(fine) if fine else None
    return fine, fam


def classify_arret(segments: list[dict], mode: str) -> dict:
    """segments = [{'chef', 'texte_segment', 'visa_cite'}, ...].

    Returns the set of families, the fine ground per chef, and the chef count.
    """
    per_chef = []
    fams = set()
    for seg in segments:
        fine, fam = classify_segment(seg.get("texte_segment", ""),
                                     seg.get("visa_cite", ""), mode)
        per_chef.append({"chef": seg.get("chef", ""), "fine": fine, "famille": fam,
                         "visa_cite": seg.get("visa_cite", "")})
        if fam:
            fams.add(fam)
    return {"familles": fams, "n_chefs": len(segments), "par_chef": per_chef}

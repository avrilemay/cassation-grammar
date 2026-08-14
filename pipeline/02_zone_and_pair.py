#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 of the pipeline: splits each ruling into zones and pairs each ground with its reasons.

Reads the cleaned corpus (five civil chambers, 2016 to 2025) and writes one
line per (ground, reasons) pair. Zoning locates, inside the ruling's text,
the segment raising each ground and the reasons answering it. Pairing then
matches the first ground with the first reasons block, the second with the
second, and so on. Judilibre's own zone markers are sometimes missing, so a
sequence of rules (the steps named A0 to A5 in this file) completes the
pairing. No outcome and no doctrinal family are assigned here.

Usage: python3 02_zone_and_pair.py [OUT.jsonl] [--check BASELINE.jsonl]
"""

import json
import re
import html
import hashlib
import sys
from pathlib import Path
from collections import Counter, defaultdict
from time import time

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402


def load_excluded_arrets(json_path):
    """Return (excluded_ids, cassation_type_override) parsed from the fixed,
    human-reviewed reference file pipeline/reference/excluded_arrets.json: a
    set of decision ids to drop (petitions, appeals and stays of execution
    that the source metadata labels like a pourvoi), and a dict mapping a
    decision id to the full or partial cassation type the human review
    settled."""
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    excluded_ids = set(data["exclude_requetes"]) | set(data["exclude_rabats"])
    cassation_type_override = dict(data["cassation_type_override"])
    return excluded_ids, cassation_type_override


def _d(text, s, e):
    """Slice the raw text and decode it (html.unescape).

    Judilibre's zone boundaries (motiv_start, motiv_end, and so on) are
    computed on the pickle's raw text, which still has &apos;, &quot;, and
    so on. Offsets stay expressed against that raw text. Each slice is
    decoded only once it is produced. The stored motivation_text and
    moyen_text come out already clean, and the regexes that read them
    later, including the classification step, match clean text.
    """
    return html.unescape(text[s:e]) if text else ''

# Config-driven paths
_cfg = load_config()
PICKLE = resolve(_cfg["data"]["clean_pickle"])
DEFAULT_OUT = resolve(_cfg["data"]["pairs_appariement"])
EXCLUDED_ARRETS_JSON = resolve(_cfg["pairing"]["excluded_arrets_json"])

ALL_CHAMBRES = _cfg["pairing"]["chambers"]
YEAR_MIN = _cfg["pairing"]["year_min"]
YEAR_MAX = _cfg["pairing"]["year_max"]
SOLUTIONS = _cfg["pairing"]["solutions"]

# A human-reviewed reference file drives two things here: a filter dropping
# decisions that are petitions, appeals, or stay requests rather than a
# genuine pourvoi on the merits, and a correction of the cassation type for
# the decisions where the regex left it ambiguous.
EXCLUDED_ARRET_IDS, CASSATION_TYPE_OVERRIDE = load_excluded_arrets(EXCLUDED_ARRETS_JSON)

# A0, zone extraction
FLAGS = re.IGNORECASE | re.UNICODE

RE_PAR_CES_MOTIFS = re.compile(r'PAR\s+CES\s+MOTIFS', FLAGS)
RE_PART_RESTRICTIVE = re.compile(
    r"casse\s+et\s+annule.{1,150}?(?:mais\s+seulement|sauf|en\s+ses\s+seules?\s+dispositions?|en\s+sa\s+seule\s+disposition|en\s+ce\s+qu[ei\'’])",
    FLAGS | re.DOTALL,
)
RE_PART_AUTRE = re.compile(
    r'casse\s+et\s+annule.{1,150}?(?:uniquement|partiellement)',
    FLAGS | re.DOTALL,
)
RE_TOTALE_EXPLICITE = re.compile(
    r'casse\s+et\s+annule.{1,80}?(?:en\s+toutes\s+ses\s+dispositions|dans\s+toutes\s+ses\s+dispositions)',
    FLAGS | re.DOTALL,
)
RE_CASSE_ANNULE_GENERIC = re.compile(r'casse\s+et\s+annule', FLAGS)


def _safe_get_list(zones_field, key):
    if not isinstance(zones_field, dict):
        return None
    v = zones_field.get(key)
    if not isinstance(v, list) or len(v) == 0:
        return None
    return v


def _concat_spans(text, spans):
    if not text or not spans:
        return ''
    parts = []
    for span in sorted(spans, key=lambda s: s.get('start', 0)):
        a, b = span.get('start', 0), span.get('end')
        if a is None or b is None:
            continue
        parts.append(text[int(a):int(b)])
    return '\n'.join(parts).strip()


def extract_dispositif(text, zones):
    spans = _safe_get_list(zones, 'dispositif')
    if spans is not None:
        d = _concat_spans(text or '', spans)
        if d:
            return d
    if not text:
        return ''
    m = RE_PAR_CES_MOTIFS.search(text)
    if m:
        return text[m.start():]
    return ''


def classify_cassation_type(dispositif):
    if not dispositif:
        return 'ambigu'
    if RE_PART_RESTRICTIVE.search(dispositif):
        return 'partielle'
    if RE_PART_AUTRE.search(dispositif):
        return 'partielle'
    if RE_TOTALE_EXPLICITE.search(dispositif):
        return 'totale'
    if RE_CASSE_ANNULE_GENERIC.search(dispositif):
        return 'totale'
    return 'ambigu'


def _sorted_zones(zlist):
    out = []
    for z in zlist or []:
        if not isinstance(z, dict):
            continue
        s, e = z.get('start'), z.get('end')
        if isinstance(s, int) and isinstance(e, int) and s < e:
            out.append((s, e))
    out.sort(key=lambda x: x[0])
    return out


# A1, detection of grounds raised sua sponte ("relevé d'office")
# Looks in the first 600 characters of the reasons text (not a strict
# header) for: "relevé d'office" in any form, "après avis donné aux
# parties" (a formula typically used for a ground raised sua sponte), or
# "article 1015 du code de procédure civile" (the streamlined non-admission
# procedure, or a ground raised sua sponte).
RE_RELEVE_OFFICE_BROAD = re.compile(
    r"(?:relev[ée]\s+d['’]\s*office"
    r"|apr[èe]s\s+avis\s+(?:donn[ée])?\s*(?:aux\s+parties)?"
    r"|article\s+1015\s+du\s+code\s+de\s+proc[ée]dure\s+civile)",
    re.IGNORECASE,
)


def is_releve_office_motiv(motiv_text):
    if not motiv_text:
        return False
    head = motiv_text[:600]
    return bool(RE_RELEVE_OFFICE_BROAD.search(head))


# A2, naive pairing
RE_VOIE_CONSEQUENCE = re.compile(r'par\s+voie\s+de\s+cons[ée]quence', re.IGNORECASE)


def pair_moyens_motivations(text, zones):
    if not isinstance(zones, dict):
        return {'pairs': [], 'appariement_ok': False, 'nsam_bloc': False, 'reason_excl': 'pas_de_zones'}
    if len(zones) == 0:
        return {'pairs': [], 'appariement_ok': False, 'nsam_bloc': False, 'reason_excl': 'zones_vides'}
    moyens = _sorted_zones(zones.get('moyens'))
    motivs = _sorted_zones(zones.get('motivations'))
    if len(moyens) == 0 and len(motivs) >= 1:
        mstart = motivs[0][0]
        mend = motivs[-1][1]
        mtext = _d(text, mstart, mend)
        pair = {
            'moyen_idx': 0, 'moyen_text': '', 'motivation_text': mtext,
            'moyen_start': -1, 'moyen_end': -1,
            'motiv_start': mstart, 'motiv_end': mend,
            'flag_nsam_bloc': True,
            'flag_motivation_vide': (len(mtext.strip()) < 50) or bool(RE_VOIE_CONSEQUENCE.search(mtext[:300])),
        }
        return {'pairs': [pair], 'appariement_ok': True, 'nsam_bloc': True, 'reason_excl': None}
    if len(moyens) == 0 and len(motivs) == 0:
        return {'pairs': [], 'appariement_ok': False, 'nsam_bloc': False, 'reason_excl': 'aucun_moyen_ni_motivation'}
    if len(moyens) >= 1 and len(motivs) == 0:
        return {'pairs': [], 'appariement_ok': False, 'nsam_bloc': False, 'reason_excl': 'moyens_sans_motivation'}
    used_motiv = [False] * len(motivs)
    pairs = []
    for i, (ms, me) in enumerate(moyens):
        chosen = -1
        for j, (mts, mte) in enumerate(motivs):
            if used_motiv[j]:
                continue
            if mts >= me:
                chosen = j
                break
        if chosen == -1:
            continue
        used_motiv[chosen] = True
        mts, mte = motivs[chosen]
        moyen_text = _d(text, ms, me)
        motiv_text = _d(text, mts, mte)
        head = motiv_text[:300]
        pair = {
            'moyen_idx': i, 'moyen_text': moyen_text, 'motivation_text': motiv_text,
            'moyen_start': ms, 'moyen_end': me,
            'motiv_start': mts, 'motiv_end': mte,
            'flag_nsam_bloc': False,
            'flag_motivation_vide': (len(motiv_text.strip()) < 50) or bool(RE_VOIE_CONSEQUENCE.search(head)),
        }
        pairs.append(pair)
    appariement_ok = (len(pairs) == len(moyens)) and (len(moyens) == len(motivs))
    return {'pairs': pairs, 'appariement_ok': appariement_ok, 'nsam_bloc': False,
            'reason_excl': None if pairs else 'aucun_appariement_possible'}


# A3, structural rules (the split-reasons case is handled separately below)
RE_PVC = re.compile(r'par\s+voie\s+de\s+cons[ée]quence', re.IGNORECASE)
RE_624 = re.compile(r'(?:article\s+|art\.?\s*)624\b', re.IGNORECASE)
RE_LIEN_DEP = re.compile(r'lien\s+de\s+d[ée]pendance\s+n[ée]cessaire', re.IGNORECASE)
RE_NSAM = re.compile(r'manifestement\s+pas\s+de\s+nature\s+(?:à|a)\s+entra[îi]ner\s+la\s+cassation', re.IGNORECASE)

RE_MOYEN_HEAD_ORDINAL = re.compile(
    r'(?:\bsur\s+(?:le|les)\s+)?'
    r'(premier|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|sixi[èe]me|'
    r'septi[èe]me|huiti[èe]me|second(?:e)?|unique)'
    r'(?:\s+et\s+(?:le|les)\s+(?:deuxi[èe]me|troisi[èe]me|quatri[èe]me|second(?:e)?))?'
    r'\s+moyens?',
    re.IGNORECASE,
)
RE_REUNIS = re.compile(r'r[ée]unis\b', re.IGNORECASE)

RE_M0_LABEL_ORDINAL = re.compile(
    r'\bsur\s+(?:le|les)\s+'
    r'(?:premier|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|sixi[èe]me|'
    r'septi[èe]me|huiti[èe]me|second(?:e)?|seul|deux|trois)'
    r'(?:\s+et\s+(?:le|les)\s+(?:deuxi[èe]me|troisi[èe]me|quatri[èe]me|second(?:e)?))?'
    r'\s+moyens?',
    re.IGNORECASE,
)
RE_M0_LABEL_INVERTED = re.compile(r'\bsur\s+(?:le|les)\s+moyens?\s+(?:unique|seul)\b', re.IGNORECASE)
RE_M0_LABEL_REUNIS = re.compile(r'\bmoyens?\s+r[ée]unis\b', re.IGNORECASE)
RE_M1_GRIEF_HEAD = re.compile(
    r'^\s*(?:Attendu\s+(?:que|,)|Mais\s+attendu|fait\s+grief|reproche\s+à)',
    re.IGNORECASE,
)


def is_pvc_text(s):
    return bool(RE_PVC.search(s) or RE_624.search(s) or RE_LIEN_DEP.search(s))


def head_token(text):
    head = text[:300]
    m = RE_MOYEN_HEAD_ORDINAL.search(head)
    num = m.group(1).lower() if m else None
    reunis = bool(RE_REUNIS.search(head))
    return num, reunis


def detect_rule(text, moyens, motivs):
    """Chooses how to reconcile the number of grounds and reasons blocks
    found in this ruling, when naive pairing has left some of them
    unmatched.

    Does not handle the case where a reasons block was split into two
    segments by a short concluding formula: that case is resolved
    afterward, in a separate pass (see motiv_scindee_trigger).
    """
    n_moy, n_mot = len(moyens), len(motivs)
    delta = n_mot - n_moy
    detail = {'n_moyens': n_moy, 'n_motiv': n_mot, 'delta': delta}
    if delta == 0:
        return 'ok', detail
    if delta >= 1 and n_moy >= 1 and n_mot >= 1:
        n_pre = 0
        for s, e in motivs:
            if s < moyens[0][0]:
                n_pre += 1
            else:
                break
        n_pvc = 0
        for s, e in reversed(motivs):
            if s < moyens[-1][1]:
                break
            if is_pvc_text(_d(text, s, e)):
                n_pvc += 1
            else:
                break
        couverts = n_pre + n_pvc
        detail['n_preambule_tete'] = n_pre
        detail['n_pvc_queue'] = n_pvc
        if delta == 1:
            if n_pre >= 1:
                return 'preambule_nsam', detail
            if n_pvc >= 1:
                return 'pvc_supplementaire', detail
            return 'audit_manuel', detail
        else:
            if couverts >= delta:
                return 'pvc_et_preambule', detail
            if couverts >= 1:
                return 'pvc_et_preambule_partiel', detail
            return 'audit_manuel', detail
    if delta <= -1 and n_moy >= 2:
        n_scindes_strict = 0
        motiv_starts = [m[0] for m in motivs]
        for i in range(len(moyens) - 1):
            ms_a, me_a = moyens[i]
            ms_b, me_b = moyens[i + 1]
            between = any(me_a <= s < ms_b for s in motiv_starts)
            if between:
                continue
            num_a, ru_a = head_token(_d(text, ms_a, me_a))
            num_b, ru_b = head_token(_d(text, ms_b, me_b))
            same_num = (num_a is not None and num_a == num_b)
            any_reunis = ru_a or ru_b
            if same_num or any_reunis:
                n_scindes_strict += 1
        detail['n_pairs_scindes_strict'] = n_scindes_strict
        if n_scindes_strict >= -delta:
            return 'moyens_scindes', detail
        if n_scindes_strict >= 1:
            return 'moyens_scindes_partiel', detail
        if n_moy == 2:
            s0, e0 = moyens[0]
            s1, e1 = moyens[1]
            m0_text = _d(text, s0, e0)
            m1_head = _d(text, s1, s1 + 200)
            m0_short = (e0 - s0) < 250
            m0_label = bool(
                RE_M0_LABEL_ORDINAL.search(m0_text)
                or RE_M0_LABEL_INVERTED.search(m0_text)
                or RE_M0_LABEL_REUNIS.search(m0_text)
            )
            m1_grief = bool(RE_M1_GRIEF_HEAD.search(m1_head)) or 'fait grief' in m1_head[:200].lower()
            if m0_short and m0_label and m1_grief:
                return 'moyen_unique_scinde', detail
        return 'audit_manuel', detail
    return 'audit_manuel', detail


# A3 helpers, applying the structural rules
def rec_base(p, **overrides):
    out = dict(p)
    for k in ('flag_pvc', 'flag_nsam_groupe_preambule', 'flag_moyens_scindes',
              'flag_motiv_scindee', 'flag_moyen_unique_scinde', 'flag_releve_office'):
        out.setdefault(k, False)
    out.setdefault('reappar_rule', None)
    out.update(overrides)
    return out


def apply_rule(rule, detail, ps, text, moyens, motivs, common):
    """Builds the output pairs for this decision, given the rule chosen by detect_rule.

    common holds the fields shared by every pair: arret_id, chambre, annee,
    decision_date, pourvoi, solution, cassation_type, n_moyens_arret.
    """
    out = []

    if rule == 'ok':
        for p in ps:
            out.append(rec_base(p, appariement_ok=True))
        return out

    if rule in ('pvc_supplementaire', 'pvc_et_preambule', 'pvc_et_preambule_partiel'):
        for p in ps:
            out.append(rec_base(p, appariement_ok=True, reappar_rule=rule))
        n_pvc = detail.get('n_pvc_queue', 1 if rule == 'pvc_supplementaire' else 0)
        last_moyen_idx = max((p['moyen_idx'] for p in ps), default=0)
        last_moyen = moyens[-1] if moyens else None
        for k in range(n_pvc):
            ms_idx = len(motivs) - 1 - k
            if ms_idx < 0:
                break
            ms, me = motivs[ms_idx]
            mtxt = _d(text, ms, me)
            last_moyen_idx += 1
            pair_seed = {
                'moyen_idx': last_moyen_idx,
                'moyen_text': _d(text, last_moyen[0], last_moyen[1]) if last_moyen else '',
                'motivation_text': mtxt,
                'moyen_start': last_moyen[0] if last_moyen else -1,
                'moyen_end': last_moyen[1] if last_moyen else -1,
                'motiv_start': ms, 'motiv_end': me,
                'flag_nsam_bloc': False,
                'flag_motivation_vide': len(mtxt.strip()) < 50,
            }
            pair_seed.update(common)
            out.append(rec_base(pair_seed, appariement_ok=True, flag_pvc=True, reappar_rule=rule))
        n_pre = detail.get('n_preambule_tete', 0)
        for k in range(n_pre):
            ms, me = motivs[k]
            ptxt = _d(text, ms, me)
            pair_seed = {
                'moyen_idx': -1 - k,
                'moyen_text': '', 'motivation_text': ptxt,
                'moyen_start': -1, 'moyen_end': -1,
                'motiv_start': ms, 'motiv_end': me,
                'flag_nsam_bloc': False,
                'flag_motivation_vide': len(ptxt.strip()) < 50,
            }
            pair_seed.update(common)
            out.append(rec_base(pair_seed, appariement_ok=True, flag_nsam_groupe_preambule=True, reappar_rule=rule))
        return out

    if rule == 'preambule_nsam':
        for p in ps:
            out.append(rec_base(p, appariement_ok=True, reappar_rule=rule))
        ms, me = motivs[0]
        ptxt = _d(text, ms, me)
        pair_seed = {
            'moyen_idx': -1, 'moyen_text': '', 'motivation_text': ptxt,
            'moyen_start': -1, 'moyen_end': -1,
            'motiv_start': ms, 'motiv_end': me,
            'flag_nsam_bloc': False,
            'flag_motivation_vide': len(ptxt.strip()) < 50,
        }
        pair_seed.update(common)
        out.append(rec_base(pair_seed, appariement_ok=True, flag_nsam_groupe_preambule=True, reappar_rule=rule))
        return out

    if rule in ('moyens_scindes', 'moyens_scindes_partiel'):
        for p in ps:
            out.append(rec_base(p, appariement_ok=True, flag_moyens_scindes=True, reappar_rule=rule))
        return out

    if rule == 'moyen_unique_scinde':
        ms_start = moyens[0][0]
        ms_end = moyens[-1][1]
        concat_moyen = _d(text, ms_start, ms_end)
        for i, p in enumerate(ps):
            if i == 0:
                out.append(rec_base(
                    p, appariement_ok=True,
                    moyen_text=concat_moyen,
                    moyen_start=ms_start, moyen_end=ms_end,
                    flag_moyen_unique_scinde=True,
                    reappar_rule=rule,
                ))
            else:
                out.append(rec_base(p, appariement_ok=True, reappar_rule=rule))
        return out

    # Pairs set aside for human audit (recorded under the audit_manuel value)
    for p in ps:
        out.append(rec_base(p, appariement_ok=False, reappar_rule='audit_manuel'))
    return out


# A3, split-reasons handling (a separate pass, run after detect_rule)
RE_TERM_INTRO = re.compile(
    r"Qu['’]en\s+(?:statuant|se\s+d[ée]terminant|d[ée]cidant)\s+ainsi",
    re.IGNORECASE,
)
RE_GRIEF_TERMINAL = re.compile(
    r"(?:a\s+|ont\s+)?(?:viol[ée]|m[ée]connu)|"
    r"(?:n['’]a\s+pas\s+donn[ée]\s+de\s+base\s+l[ée]gale|priv[ée]?\s+sa\s+d[ée]cision\s+de\s+base\s+l[ée]gale)|"
    r"exc[èe]s\s+de\s+pouvoir|d[ée]natur[ée]?|"
    r"d[ée]faut\s+de\s+r[ée]ponse\s+(?:à\s+|aux\s+)?conclusions?",
    re.IGNORECASE,
)
# Secondary branch: a short concluding violation formula, without requiring the
# terminal formula "Qu'en statuant ainsi".
RE_CONCL_VIOLE_SUSVISE = re.compile(
    r"viol[ée]\s+(?:les?\s+)?(?:textes?|articles?)\s+susvis[ée]s?",
    re.IGNORECASE,
)


def has_terminal_with_grief(motiv_text):
    if not motiv_text:
        return False
    m = RE_TERM_INTRO.search(motiv_text)
    if not m:
        return False
    window = motiv_text[m.start(): m.start() + 1000]
    return bool(RE_GRIEF_TERMINAL.search(window))


def motiv_scindee_trigger(motiv_text):
    """Decides whether a reasons block was split into two segments that
    belong back together.

    Two ways to trigger. Branch A: the block ends with the closing formula
    "Qu'en statuant..." followed by a grief verb. Branch B: the block is
    short (under 800 characters) and ends with "violé les textes/articles
    susvisés". Returns (triggered, branch). Branch A wins when both match.
    """
    if not motiv_text:
        return False, None
    if has_terminal_with_grief(motiv_text):
        return True, 'terminal_grief'
    if len(motiv_text) < 800 and RE_CONCL_VIOLE_SUSVISE.search(motiv_text):
        return True, 'short_viole_susvise'
    return False, None


def apply_motiv_scindee_unified(by_arret, zones_by_aid):
    """For each decision, merges the last reasons segment onto the ground it
    belongs to when three conditions hold: the segment triggers
    motiv_scindee_trigger, it is not already covered by an existing pair,
    and the second-to-last segment is already paired to a ground, the merge
    target. The triggering branch is kept in motiv_scindee_branch for
    diagnostics.
    """
    n_applied = 0
    n_by_branch = Counter()
    for aid, ps in by_arret.items():
        zinfo = zones_by_aid.get(aid)
        if zinfo is None:
            continue
        text, moyens, motivs = zinfo
        if not motivs or len(motivs) < 2:
            continue
        last_s, last_e = motivs[-1]
        # Skip if already covered
        if any(p.get('motiv_start', -1) <= last_s and p.get('motiv_end', -1) >= last_e for p in ps):
            continue
        last_text = _d(text, last_s, last_e)
        triggered, branch = motiv_scindee_trigger(last_text)
        if not triggered:
            continue
        prev_s, prev_e = motivs[-2]
        target_idx = None
        for i, p in enumerate(ps):
            if p.get('motiv_start', -1) == prev_s and p.get('motiv_end', -1) == prev_e:
                target_idx = i
                break
        if target_idx is None:
            continue
        p_old = ps[target_idx]
        new_motiv_text = _d(text, p_old['motiv_start'], last_e)
        p_new = dict(p_old)
        p_new['motiv_end'] = last_e
        p_new['motivation_text'] = new_motiv_text
        p_new['reappar_rule'] = 'motiv_scindee'
        p_new['motiv_scindee_branch'] = branch
        p_new['flag_motiv_scindee'] = True
        p_new['appariement_ok'] = True
        p_new['flag_motivation_vide'] = len(new_motiv_text.strip()) < 50
        ps[target_idx] = p_new
        n_applied += 1
        n_by_branch[branch] += 1
    return n_applied, n_by_branch


# A4, gate_terminal_recovery (a visa, or a full quashing plus "CASSE ET ANNULE")
VISA_RE = re.compile(r'^\s*Vu\b[^.\n]{0,200}\bl[\' ]?articles?\b', re.IGNORECASE)
VISA_RE_LOOSE = re.compile(r'(?:\n|^)\s*Vu[^.\n]{1,30}\bl[\' ]?articles?\b', re.IGNORECASE)
CASSE_RE = re.compile(r'\bCASSE\s+ET\s+ANNULE\b', re.IGNORECASE)


def gate_A(text, motivs):
    """True if a visa is present in the last reasons segment or the one before it."""
    if not motivs:
        return False
    n_mot = len(motivs)
    s, e = motivs[-1]
    orphan_text = _d(text, s, e)
    head_orphan = orphan_text[:300]
    if VISA_RE.search(head_orphan) or VISA_RE_LOOSE.search('\n' + head_orphan):
        return True
    if n_mot >= 2:
        ps, pe = motivs[-2]
        tail_prev = _d(text, max(ps, pe - 400), pe)
        if VISA_RE_LOOSE.search('\n' + tail_prev) or VISA_RE.search(tail_prev):
            return True
        inter = _d(text, pe, s)
        if 0 < len(inter) < 500:
            if VISA_RE_LOOSE.search('\n' + inter):
                return True
    return False


def gate_B(text, dispositifs, cassation_type):
    if cassation_type != 'totale':
        return False
    if not dispositifs:
        return False
    for s, e in dispositifs:
        if CASSE_RE.search(_d(text, s, e)):
            return True
    return False


def apply_gate_terminal_recovery(by_arret, zones_by_aid):
    """For each decision not already handled by the split-reasons pass,
    merges an orphaned last reasons segment onto the pair right before it
    (target.motiv_end at most orphan.start), when either gate fires: gate A,
    a visa is present, or gate B, a full quashing plus the closing formula
    "CASSE ET ANNULE"."""
    n_visa = 0
    n_disp = 0
    for aid, ps in by_arret.items():
        zinfo = zones_by_aid.get(aid)
        if zinfo is None:
            continue
        text, moyens, motivs, dispositifs, cassation_type = zinfo['text'], zinfo['moyens'], zinfo['motivs'], zinfo['dispositifs'], zinfo['cassation_type']
        if not motivs or len(motivs) < 2:
            continue
        last_s, last_e = motivs[-1]
        # Skip if already covered
        if any(p.get('motiv_start', -1) <= last_s and p.get('motiv_end', -1) >= last_e for p in ps):
            continue
        # Skip pairs already handled by the split-reasons step above (idempotent)
        if any(p.get('reappar_rule') == 'motiv_scindee' for p in ps):
            continue
        gA = gate_A(text, motivs)
        gB = gate_B(text, dispositifs, cassation_type)
        if not (gA or gB):
            continue
        # Target: the last pair whose reasons end before the orphan block starts
        cands = [(i, p) for i, p in enumerate(ps) if p.get('motiv_end', -1) <= last_s and p.get('motiv_end', -1) > -1]
        if not cands:
            continue
        target_idx, target = max(cands, key=lambda x: x[1].get('motiv_end', -1))
        new_motiv_text = _d(text, target['motiv_start'], last_e)
        p_new = dict(target)
        # Diagnostic trace: reasons text and motiv_end before the gate merge
        p_new['motivation_text_pre_gate'] = target.get('motivation_text')
        p_new['motiv_end_pre_gate'] = target.get('motiv_end')
        p_new['motiv_end'] = last_e
        p_new['motivation_text'] = new_motiv_text
        p_new['reappar_rule'] = 'gate_visa' if gA else 'gate_dispositif'
        p_new['appariement_ok'] = True
        p_new['flag_motivation_vide'] = len(new_motiv_text.strip()) < 50
        ps[target_idx] = p_new
        if gA:
            n_visa += 1
        else:
            n_disp += 1
    return n_visa, n_disp


# A5, creation of pairs for grounds raised sua sponte
def apply_releve_office_creation(by_arret, zones_by_aid):
    """For each decision, creates a synthetic pair (with a negative ground
    index) for each reasons segment flagged by A1 as raised sua sponte that
    is not already covered by a pair produced by A2, A3, or A4."""
    n_created = 0
    for aid, ps in by_arret.items():
        zinfo = zones_by_aid.get(aid)
        if zinfo is None:
            continue
        text = zinfo['text']
        motivs = zinfo['motivs']
        if not motivs:
            continue

        chambre = ps[0]['chambre'] if ps else None
        annee = ps[0]['annee'] if ps else None
        decision_date = ps[0]['decision_date'] if ps else None
        pourvoi = ps[0]['pourvoi'] if ps else None
        solution = ps[0]['solution'] if ps else None
        cassation_type = ps[0]['cassation_type'] if ps else None
        n_moyens_arret = ps[0]['n_moyens_arret'] if ps else 0

        # Find the lowest negative moyen_idx already used (preambule_nsam gives -1, -2, ...)
        used_neg = [p['moyen_idx'] for p in ps if p['moyen_idx'] < 0]
        next_neg = min(used_neg) - 1 if used_neg else -2

        k = 0
        for s, e in motivs:
            mtxt = _d(text, s, e)
            if not is_releve_office_motiv(mtxt):
                continue
            # Skip if already strictly covered by an existing pair
            if any(p.get('motiv_start', -1) <= s and p.get('motiv_end', -1) >= e for p in ps):
                continue
            pair = {
                'arret_id': aid, 'chambre': chambre, 'annee': annee,
                'decision_date': decision_date, 'pourvoi': pourvoi,
                'solution': solution, 'cassation_type': cassation_type,
                'moyen_idx': next_neg - k,
                'moyen_text': '',
                'motivation_text': mtxt,
                'moyen_start': -1, 'moyen_end': -1,
                'motiv_start': s, 'motiv_end': e,
                'n_moyens_arret': n_moyens_arret,
                'flag_nsam_bloc': False,
                'flag_motivation_vide': len(mtxt.strip()) < 50,
            }
            ps.append(rec_base(pair, appariement_ok=True, flag_releve_office=True, reappar_rule='releve_office'))
            k += 1
            n_created += 1
    return n_created


# Main pipeline
def run_appariement(out_path):
    OUT_JSONL = Path(out_path)
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    t_total = time()
    print('=== step 2: zoning and pairing ===')
    print(f'Output: {OUT_JSONL}')

    # Load the corpus and scope it to the five civil chambers
    t = time()
    df_full = pd.read_pickle(PICKLE)
    df_full['annee'] = pd.to_datetime(df_full['decision_date'], errors='coerce').dt.year
    USE_COLS = ['id', 'chamber', 'annee', 'solution', 'decision_date', 'number', 'text', 'zones']
    df = df_full[USE_COLS].copy()
    mask = (
        df['chamber'].isin(ALL_CHAMBRES)
        & (df['annee'] >= YEAR_MIN) & (df['annee'] <= YEAR_MAX)
        & df['solution'].isin(SOLUTIONS)
    )
    df = df.loc[mask].reset_index(drop=True)
    print(f'  scope (5 chambers): {len(df):,} decisions ({time()-t:.1f}s)')

    # Drop petitions, appeals, and stay requests: a human-reviewed sample
    # flags decisions that are not a genuine pourvoi on the merits
    n_before = len(df)
    df = df.loc[~df['id'].isin(EXCLUDED_ARRET_IDS)].reset_index(drop=True)
    print(f'  excluded_arrets filter: -{n_before - len(df):,} decisions (petitions/appeals/stays) -> {len(df):,}')

    # Determine cassation_type for each decision
    # html.unescape on the operative part to correctly match "casse et annule... qu['']"
    # when the pickle contains &apos; instead of '.
    df['cassation_type'] = [
        classify_cassation_type(html.unescape(extract_dispositif(t, z)))
        for t, z in zip(df['text'], df['zones'])
    ]
    # Human override of cassation_type, for decisions the regex left ambiguous
    if CASSATION_TYPE_OVERRIDE:
        ov = df['id'].map(CASSATION_TYPE_OVERRIDE)
        n_ov = int(ov.notna().sum())
        df.loc[ov.notna(), 'cassation_type'] = ov[ov.notna()]
        print(f'  cassation_type corrected from the human-reviewed file: {n_ov} decisions')

    # A2, naive pairing and zone extraction
    t = time()
    by_arret = defaultdict(list)
    zones_by_aid = {}
    arrets_flag = set()

    for row in df.itertuples(index=False):
        zones = row.zones if isinstance(row.zones, dict) else {}
        moyens = _sorted_zones(zones.get('moyens'))
        motivs = _sorted_zones(zones.get('motivations'))
        dispositifs = _sorted_zones(zones.get('dispositif'))

        zones_by_aid[row.id] = {
            'text': row.text or '',
            'moyens': moyens,
            'motivs': motivs,
            'dispositifs': dispositifs,
            'cassation_type': row.cassation_type,
        }

        out = pair_moyens_motivations(row.text or '', row.zones)
        if not out['pairs']:
            continue
        n_moyens_total = len(moyens)
        common_pair_fields = {
            'arret_id': row.id, 'chambre': row.chamber, 'annee': int(row.annee),
            'decision_date': str(row.decision_date) if row.decision_date is not None else None,
            'pourvoi': row.number, 'solution': row.solution,
            'cassation_type': row.cassation_type, 'n_moyens_arret': n_moyens_total,
        }
        for p in out['pairs']:
            rec = {
                **common_pair_fields,
                'moyen_idx': p['moyen_idx'],
                'moyen_text': p['moyen_text'], 'motivation_text': p['motivation_text'],
                'moyen_start': p['moyen_start'], 'moyen_end': p['moyen_end'],
                'motiv_start': p['motiv_start'], 'motiv_end': p['motiv_end'],
                'flag_nsam_bloc': bool(p['flag_nsam_bloc']),
                'flag_motivation_vide': bool(p['flag_motivation_vide']),
            }
            by_arret[row.id].append(rec)
        if not out['appariement_ok']:
            arrets_flag.add(row.id)
    print(f'  A2, naive pairing: {sum(len(v) for v in by_arret.values()):,} pairs, {len(arrets_flag):,} decisions flagged ({time()-t:.1f}s)')

    # A3, structural rules (detect_rule and apply_rule)
    t = time()
    counter_rule = Counter()
    by_arret_v3 = defaultdict(list)
    for aid, ps in by_arret.items():
        if aid not in arrets_flag:
            for p in ps:
                by_arret_v3[aid].append(rec_base(p, appariement_ok=True))
            counter_rule['ok'] += 1
            continue
        text = zones_by_aid[aid]['text']
        moyens = zones_by_aid[aid]['moyens']
        motivs = zones_by_aid[aid]['motivs']
        rule, detail = detect_rule(text, moyens, motivs)
        counter_rule[rule] += 1
        common = {k: ps[0][k] for k in ('arret_id', 'chambre', 'annee', 'decision_date', 'pourvoi', 'solution', 'cassation_type', 'n_moyens_arret')}
        out = apply_rule(rule, detail, ps, text, moyens, motivs, common)
        by_arret_v3[aid] = out
    print(f'  A3, structural rules: {sum(len(v) for v in by_arret_v3.values()):,} pairs (top rules: {dict(counter_rule.most_common(10))}) ({time()-t:.1f}s)')

    # A3bis, split-reasons handling
    t = time()
    zones_min = {aid: (z['text'], z['moyens'], z['motivs']) for aid, z in zones_by_aid.items()}
    n_motiv_scindee, branch_counts = apply_motiv_scindee_unified(by_arret_v3, zones_min)
    print(f'  A3, split-grounds-text applied: {n_motiv_scindee} decisions '
          f'(branches: {dict(branch_counts)}) ({time()-t:.1f}s)')

    # A4, gate_terminal_recovery
    t = time()
    n_visa, n_disp = apply_gate_terminal_recovery(by_arret_v3, zones_by_aid)
    print(f'  A4, gate_visa: {n_visa}  gate_dispositif: {n_disp}  total: {n_visa + n_disp} ({time()-t:.1f}s)')

    # A5, pairs created for grounds raised sua sponte
    t = time()
    n_releve = apply_releve_office_creation(by_arret_v3, zones_by_aid)
    print(f'  A5, sua-sponte pairs created: {n_releve} ({time()-t:.1f}s)')

    # Write output
    t = time()
    all_pairs = []
    for aid, ps in by_arret_v3.items():
        ps.sort(key=lambda p: (p['moyen_idx'] if p['moyen_idx'] >= 0 else 1000 + abs(p['moyen_idx'])))
        all_pairs.extend(ps)

    h = hashlib.sha256()
    with OUT_JSONL.open('w', encoding='utf-8') as fh:
        for rec in all_pairs:
            line = json.dumps(rec, ensure_ascii=False)
            fh.write(line)
            fh.write('\n')
            h.update(line.encode('utf-8'))
            h.update(b'\n')
    sha = h.hexdigest()
    print(f'  Written: {OUT_JSONL.name} ({len(all_pairs):,} pairs, sha {sha[:16]}...) ({time()-t:.1f}s)')

    # Summary
    print()
    print('=== Step 2 summary ===')
    print(f'  Total pairs: {len(all_pairs):,}')
    print(f'  Decisions covered: {len(by_arret_v3):,}')
    rr_counts = Counter()
    flag_counts = Counter()
    for p in all_pairs:
        rr = p.get('reappar_rule')
        if rr:
            rr_counts[rr] += 1
        for f in ('flag_pvc', 'flag_nsam_groupe_preambule', 'flag_moyens_scindes',
                  'flag_motiv_scindee', 'flag_moyen_unique_scinde', 'flag_releve_office',
                  'flag_nsam_bloc'):
            if p.get(f):
                flag_counts[f] += 1
    print('  reappar_rule counts:')
    for k, v in sorted(rr_counts.items(), key=lambda x: -x[1]):
        print(f'    {k:35s} {v:6d}')
    print('  flags:')
    for k, v in sorted(flag_counts.items(), key=lambda x: -x[1]):
        print(f'    {k:35s} {v:6d}')

    print()
    print(f'Total time: {time()-t_total:.1f}s')
    return sha


def main():
    argv = sys.argv[1:]
    check = None
    if '--check' in argv:
        i = argv.index('--check')
        check = Path(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    out = Path(argv[0]) if argv else DEFAULT_OUT
    sha = run_appariement(out)

    if check is not None:
        print(f'\n=== non-regression check vs {check.name} ===')
        diffs = Counter()
        n = 0
        with open(out, encoding='utf-8') as fa, open(check, encoding='utf-8') as fb:
            for la, lb in zip(fa, fb):
                if not la.strip() and not lb.strip():
                    continue
                a = json.loads(la)
                b = json.loads(lb)
                n += 1
                for k in set(a) | set(b):
                    if a.get(k) != b.get(k):
                        diffs[k] += 1
        # length check (keys (arret_id, moyen_idx) aligned line by line)
        na = sum(1 for l in open(out, encoding='utf-8') if l.strip())
        nb_ = sum(1 for l in open(check, encoding='utf-8') if l.strip())
        print(f'pairs: {n:,}  (out={na:,}  baseline={nb_:,})')
        if na != nb_:
            print(f'MISMATCHED LINE COUNT: out={na} baseline={nb_}')
            sys.exit(1)
        if diffs:
            print('DIFFS:', dict(diffs))
            sys.exit(1)
        print('0 DIFF: 02_zone_and_pair.py reproduces the baseline exactly.')


if __name__ == '__main__':
    main()

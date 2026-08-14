"""Classifies a pair's reasons text and derives its final status.

For a rejected ground, this module scans the reasons text against the grid
(named patterns from grids/grille.json, grouped into r-code families) to
find which family's closing formula applies. For an accepted ground, it
calls the cascade (the fixed-order regex families in grids/detectors.py) to
get the fine-grained ground and doctrinal family. When the two signals
disagree, a conflict policy decides between them rather than forcing an
answer. This module defines no entry point. See pipeline/03_classify.py.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _config import load_config, resolve  # noqa: E402

_cfg = load_config()
HERE = Path(__file__).resolve().parent

sys.path.insert(0, str(resolve(_cfg["classify"]["detectors_dir"])))
import hybrid_lib as Hmod  # noqa: E402  (segment-level cascade)
from hors_moyen_rule import is_hors_moyen  # noqa: E402

GRILLE = resolve(_cfg["classify"]["grille_json"])

# Grid preparation and r-code scanning
with GRILLE.open(encoding="utf-8") as f:
    G = json.load(f)

PATTERNS_COMPILED = {}
for name, spec in G['patterns'].items():
    flags = re.IGNORECASE
    if spec.get('flags') == 'MULTILINE':
        flags |= re.MULTILINE
    PATTERNS_COMPILED[name] = re.compile(spec['regex'], flags)

FAM_AGG = G['family_aggregation']
PATTERN_TO_FAM = {}
for fam, names in FAM_AGG.items():
    for n in names:
        if n not in PATTERN_TO_FAM or fam not in ('R8a', 'R8b', 'R8c'):
            PATTERN_TO_FAM.setdefault(n, fam)
            if fam not in ('R8a', 'R8b', 'R8c'):
                PATTERN_TO_FAM[n] = fam

R8B_LJ_PATTERN = re.compile(r'l[ée]galement\s+justifi[ée]?\s+(?:sa|la|leur|ses)\s+d[ée]cision', re.IGNORECASE)
R8B_NEG_BEFORE = re.compile(r"n['’]?\s*(?:a|ont|y\s+ai|est|aurait|auraient|ai|aurions|auriez)\s+(?:donc\s+|en\s+rien\s+|aucunement\s+|,\s*[^,]{1,50},\s+)?pas\b", re.IGNORECASE)
# R3_MARKERS gives position_arbitrage the markers it uses to locate a
# rejection's closing formula ("n'est donc/dès lors pas fondé", "ne
# peut/peuvent être accueilli", and their plural forms). Without this, a
# false positive of substantive acceptance ("violation des dispositions")
# could wrongly flip a clear rejection towards accepte.
R3_MARKERS = re.compile(
    r"d['’]o[ùu]\s+il\s+suit"
    r"|(?:n['’]est|n['’]ont|ne\s+sont)\s+(?:(?:donc|d[èe]s\s+lors|pour\s+autant|en\s+cons[ée]quence|dans\s+ces\s+conditions)\s+){0,2}pas\s+fond[ée]s?"
    r"|moyen\s+n['’]est"
    r"|ne\s+(?:peut|peuvent|saurait|sauraient)\s+(?:donc\s+|d[èe]s\s+lors\s+)?(?:être|etre)\s+accueilli(?:s|e|es)?"
    r"|n['’]est\s+fond[ée]\s+en\s+aucune"
    r"|ne\s+(?:peut|peuvent|saurait|sauraient)\s+(?:donc\s+|d[èe]s\s+lors\s+)?prosp[ée]rer"
    r"|n['’]encour(?:t|ent)\s+pas\s+(?:les?\s+|de\s+)?(?:griefs?|critiques?|reproches?)"
    r"|sans\s+encourir\s+(?:les?\s+)?griefs?"
    r"|ne\s+donne\s+pas\s+ouverture\s+[àa]\s+(?:la\s+)?cassation"
    r"|(?<!pas\s)l[ée]galement\s+justifi[ée]", re.IGNORECASE)
# For the R8b -> R8c proximity test, the "légalement justifié" alternative is
# excluded: it would match at distance 0 from R8B_LJ_PATTERN and make "close"
# always true, destroying the point of R8c (a legally-justified closing
# formula plus a distinct nearby rejection closure).
R3_MARKERS_R8C = re.compile(
    r"d['’]o[ùu]\s+il\s+suit"
    r"|(?:n['’]est|n['’]ont|ne\s+sont)\s+(?:(?:donc|d[èe]s\s+lors|pour\s+autant|en\s+cons[ée]quence|dans\s+ces\s+conditions)\s+){0,2}pas\s+fond[ée]s?"
    r"|moyen\s+n['’]est"
    r"|ne\s+(?:peut|peuvent|saurait|sauraient)\s+(?:donc\s+|d[èe]s\s+lors\s+)?(?:être|etre)\s+accueilli(?:s|e|es)?"
    r"|n['’]est\s+fond[ée]\s+en\s+aucune"
    r"|ne\s+(?:peut|peuvent|saurait|sauraient)\s+(?:donc\s+|d[èe]s\s+lors\s+)?prosp[ée]rer"
    r"|n['’]encour(?:t|ent)\s+pas\s+(?:les?\s+|de\s+)?(?:griefs?|critiques?|reproches?)"
    r"|sans\s+encourir\s+(?:les?\s+)?griefs?"
    r"|ne\s+donne\s+pas\s+ouverture\s+[àa]\s+(?:la\s+)?cassation", re.IGNORECASE)
R9_MOYEN_FORT = re.compile(r"(?:^|\.|\n)\s*\d+\s*[°)/]\s*(?:ALORS|ET\s+ALORS|QU['’])?|\bAUX\s+MOTIFS\b|fait\s+grief\s+(?:à\s+|au\s+)?l['’](?:arr[êe]t|jugement)|selon\s+le\s+moyen|reproche\s+(?:à\s+|au\s+)?l['’]?(?:arr[êe]t|jugement|d[ée]cision)|(?:Le\s+|le\s+)?moyen\s+(?:fait|f[ai]t)\s+grief\s+à|(?-i:Pour)\s+[^.;]{3,120}?,\s+l['’]arr[êe]t\s+(?:attaqu[ée]\s+)?(?:retient|rel[èe]ve|constate|[ée]nonce)", re.IGNORECASE)
R9_MOTIVATION_OK = re.compile(r'Mais\s+attendu\s+que|R[ée]ponse\s+de\s+la\s+Cour', re.IGNORECASE)

# Pattern names grouped by r-code family, used by the guards below to check
# which family a hit belongs to without a second grid lookup.
R8B_NAMES = set(FAM_AGG.get('R8b', []))
R9_NAMES = set(FAM_AGG.get('R9', []))
R8A_NAMES = set(FAM_AGG.get('R8a', []))


def scan_r_patterns(text):
    hits = []
    for name, rx in PATTERNS_COMPILED.items():
        m = rx.search(text)
        if m:
            hits.append({'pattern': name, 'family_top': PATTERN_TO_FAM.get(name), 'start': m.start()})
    return hits


def post_process_r(hits, text):
    hit_names = {h['pattern'] for h in hits}
    if hit_names & R8B_NAMES:
        lj_matches = list(R8B_LJ_PATTERN.finditer(text))
        if lj_matches:
            has_positive = False
            for m in lj_matches:
                ctx_before = text[max(0, m.start() - 80):m.start()]
                m_neg = R8B_NEG_BEFORE.search(ctx_before)
                if not (m_neg and len(ctx_before) - m_neg.end() <= 15):  # adjacent negation
                    has_positive = True
                    break
            if not has_positive:
                hit_names -= R8B_NAMES
    if hit_names & R9_NAMES:
        kept = set()
        for name in (hit_names & R9_NAMES):
            rx = PATTERNS_COMPILED[name]
            keep_this = False
            for m in rx.finditer(text):
                ctx_before = text[max(0, m.start() - 200):m.start()]
                if R9_MOYEN_FORT.search(ctx_before) and not R9_MOTIVATION_OK.search(ctx_before):
                    continue
                keep_this = True
                break
            if keep_this:
                kept.add(name)
        hit_names = (hit_names - R9_NAMES) | kept
    r8c_flag = False
    if (hit_names & R8B_NAMES) and not (hit_names & R8A_NAMES):
        lj_matches = list(R8B_LJ_PATTERN.finditer(text))
        r3_matches = list(R3_MARKERS_R8C.finditer(text))
        if lj_matches and r3_matches:
            is_close = any(abs(lj.start() - r3.start()) < 250 for lj in lj_matches for r3 in r3_matches)
            if is_close:
                r8c_flag = True
    families = set()
    for name in hit_names:
        fam = PATTERN_TO_FAM.get(name)
        if fam and fam not in ('R8a', 'R8b', 'R8c'):
            families.add(fam)
    if hit_names & R8B_NAMES:
        families.add('R8c' if r8c_flag else 'R8b')
        families.add('R8')
    if hit_names & R8A_NAMES:
        families.add('R8a')
        families.add('R8')
    return families, {'hit_names': hit_names, 'r8c': r8c_flag}


def classify_axe1(motivation_text):
    if not isinstance(motivation_text, str) or len(motivation_text.strip()) < 50:
        return None, None
    fine, fam = Hmod.classify_segment(motivation_text, '', 'B')
    return fine, fam


# Status derivation
RE_PVC_SHORT = re.compile(r'par\s+voie\s+de\s+cons[ée]quence', re.IGNORECASE)
RE_ART_624 = re.compile(r'(?:article\s+|art\.?\s*)624\b', re.IGNORECASE)

RE_ACCUEIL_MARKER = re.compile(
    r"\b(?:casse\s+et\s+annule|"
    r"viol(?:é|ée|és|ées|ation\s+(?:de|des))|"
    r"fait\s+grief\s+(?:à\s+|au\s+)?l['’](?:arr[êêe]t|jugement|d[éee]cision)|"
    r"manqu(?:é|e)\s+(?:de\s+)?(?:donn(?:é|er)\s+)?(?:de\s+)?base\s+l[ée]gale|"
    r"exc[èe]s\s+de\s+pouvoir|"
    r"d[ée]natur(?:é|ation)|"
    r"d[ée]faut\s+de\s+r[ée]ponse|"
    r"perte\s+du?\s+fondement)\b",
    re.IGNORECASE,
)

# PVC (accepted "by way of consequence") discriminator: a genuine standalone
# acceptance marker present => the "by way of consequence" mention is just a
# ricochet or citation, not the actual ground. Distinct from
# RE_ACCUEIL_MARKER (which includes the claimant's own "fait grief à
# l'arrêt" and omits the "Vu l'article" visa).
RE_PVC_ACCUEIL = re.compile(
    r"vu\s+l['’]\s*article(?!\s*1014)|vu\s+les\s+articles(?!\s*1014)|casse\s+et\s+annule|"
    r"a\s+viol[ée]e?s?\b|viol[ée]\s+(?:l['’]|les\s|le\s|ce|cet|ledit)|(?<!sans\s)(?<!ni\s)d[ée]natur(?:é|e|ation|ant)|"
    r"(?:priv[ée]?\s+(?:sa\s+d[ée]cision\s+)?de\s+|n['’]a\s+pas\s+donn[ée]\s+(?:de\s+)?|manqu[ée]?\s+de\s+)base\s+l[ée]gale|"
    r"sans\s+rechercher|sans\s+r[ée]pondre|d[ée]faut\s+de\s+r[ée]ponse|exc[èe]s\s+de\s+pouvoir|"
    r"perte\s+du?\s+fondement", re.IGNORECASE)

# Detector for "entraîne/emporte ... par voie de conséquence" (reinforces the
# PVC route when article 624 is not cited).
VERB_PVC         = re.compile(r"(entra[îi]n|emport|s['’]?\s*[ée]tend)", re.IGNORECASE)
NEG_PVC          = re.compile(r"n['’]\s*(?:entra[îi]n\w*|emport\w*|[ée]tend\w*)\s+pas\b", re.IGNORECASE)
REJET_TARGET_PVC = re.compile(r"cons[ée]quence[\s,]{0,3}(?:le\s+|du\s+|au\s+|d['’]un\s+)?rejets?\b", re.IGNORECASE)

def _entraine_pres(motiv):
    """True if the court itself pronounces a domino effect (a linking verb
    within <=60 chars before "par voie de conséquence"), excluding negation
    ("n'entraîne pas") and a rejection domino ("conséquence ... rejet")."""
    for x in RE_PVC_SHORT.finditer(motiv):
        if not VERB_PVC.search(motiv[max(0, x.start()-60):x.start()]): continue
        if NEG_PVC.search(motiv[max(0, x.start()-60):x.start()]):       continue
        if REJET_TARGET_PVC.search(motiv[x.start():x.start()+50]):       continue
        return True
    return False

# "Composite block" signature (one ground rejected as manifestly unfounded
# alongside another ground accepted on its merits, in the same pair).
RE_RNSM_MARKER = re.compile(r"manifestement\s+pas\s+de\s+nature\s+(?:à|a)\s+entra[îi]ner\s+la\s+cassation"
                            r"|article\s+1014", re.IGNORECASE)
def _composite_cassation(motiv):
    """True if both a manifestly-unfounded marker and a genuine acceptance
    marker are present in the same grounds text (multi-ground block)."""
    return bool(RE_RNSM_MARKER.search(motiv)) and bool(RE_PVC_ACCUEIL.search(motiv))


def _pvc_est_principal(motiv):
    """True if "by way of consequence" is the primary disposition (no
    standalone acceptance marker present)."""
    return bool(RE_PVC_SHORT.search(motiv)) and RE_PVC_ACCUEIL.search(motiv) is None


# Legal-basis closing formulas count as an acceptance marker only for
# block-level pairs (empty moyen_text: the block's own outcome governs,
# neutralized by policy). For pairs attributed to one specific ground, these
# formulas usually close the response to a different ground, so they are
# excluded there.
RE_ACCUEIL_BLOC_XTRA = re.compile(
    r"priv[ée]e?s?\s+(?:sa\s+d[ée]cision\s+)?de\s+base\s+l[ée]gale"
    r"|n['’]a\s+pas\s+donn[ée]\s+(?:de\s+)?base\s+l[ée]gale", re.IGNORECASE)


def position_arbitrage(text, r_codes, bloc=False):
    if 'R3' not in r_codes:
        return None
    r3_last = max((m.start() for m in R3_MARKERS.finditer(text)), default=-1)
    acc_last = max((m.start() for m in RE_ACCUEIL_MARKER.finditer(text)), default=-1)
    if bloc:
        acc_last = max(acc_last, max((m.start() for m in RE_ACCUEIL_BLOC_XTRA.finditer(text)), default=-1))
    if r3_last == -1 and acc_last == -1:
        return None
    if r3_last > acc_last:
        return 'rejete'
    if acc_last > r3_last:
        return 'accepte'
    return None


# Conflict-resolution policy (the "scope-cut" policy)
#
# It reads the block positionally instead of forcing conflicts by decision
# shape. It first cuts off post-decision tails ("Portée et conséquences",
# release from proceedings, costs), then sets aside blocks suspected of
# holding several grounds (never forced: they become a conflict). On
# single-ground blocks, an opening rejection formula (art. 1014,
# inadmissible, inoperative, guarded against a mere "soutient que"
# restatement) resolves to rejete. Otherwise, a closing acceptance formula in
# the last 260 characters resolves to accepte. Otherwise, it stays a
# conflict. On single-ground blocks the block's outcome and the ground's
# outcome coincide, so this reading is valid either way.
RE_CUT = re.compile(
    r"(?:Port[ée]e\s+et\s+cons[ée]quences"
    r"|^\s*Mise\s+hors\s+de\s+cause"
    r"|^\s*Demandes?\s+de\s+mise\s+hors\s+de\s+cause"
    r"|^\s*Sur\s+les\s+d[ée]pens"
    r"|^\s*Et\s+attendu\s+que\s+la\s+cassation\s+.{0,120}?(?:emporte|entra[îi]ne))",
    re.IGNORECASE | re.MULTILINE)

RE_ACCUEIL_FIN = re.compile(
    r"(?:a\s+viol[ée]s?(?:,?\s+par\s+(?:fausse\s+application|refus\s+d'application),?)?\s+"
    r"(?:le\s+texte|les\s+textes|l'article|les\s+articles|le\s+principe|les\s+principes|celui-ci|celle-ci|ceux-ci|ce\s+texte|cet\s+article)"
    r"|viol[ée]\s+(?:le|les)\s+(?:texte|textes|principe)s?\s+(?:et\s+principes?\s+)?susvis[ée]s?"
    r"|priv[ée]\s+(?:sa\s+d[ée]cision\s+de\s+|de\s+)base\s+l[ée]gale"
    r"|n'a\s+pas\s+donn[ée]\s+de\s+base\s+l[ée]gale"
    r"|n'a\s+pas\s+satisfait\s+aux\s+exigences"
    r"|a\s+m[ée]connu\s+les\s+exigences"
    r"|n'a\s+pas\s+tir[ée]\s+les\s+cons[ée]quences\s+l[ée]gales"
    r"|n'a\s+pas\s+mis\s+la\s+Cour\s+de\s+cassation\s+en\s+mesure\s+d'exercer\s+son\s+contr[ôo]le"
    r"|n['’]a\s+pas\s+l[ée]galement\s+justifi[ée]"
    r"|(?:a|ont)\s+d[ée]natur[ée]"
    r"|(?:a|ont)\s+exc[ée]d[ée]\s+(?:ses|leurs)\s+pouvoirs"
    r"|m[ée]connu\s+l[’']?[ée]tendue"
    r"|casse\s+et\s+annule)", re.IGNORECASE)

RE_REJET_OUVERTURE = re.compile(
    r"(?:pas\s+lieu\s+de\s+statuer\s+par\s+une\s+d[ée]cision\s+sp[ée]cialement\s+motiv[ée]e\s+sur\s+ce(?:s)?\s+moyens?"
    r"|ce\s+moyen[^.]{0,80}?n'est\s+manifestement\s+pas\s+de\s+nature"
    r"|le\s+moyen\s+est\s+(?:donc\s+)?(?:irrecevable|inop[ée]rant)"
    r"|le\s+moyen[^.]{0,120}?n'est\s+pas\s+recevable"
    r"|moyens?\s+annex[ée]s?[^.]{0,120}?pas\s+de\s+nature\s+[àa]\s+entra[îi]ner\s+la\s+cassation)",
    re.IGNORECASE)

FIN_WINDOW = 260
OUV_WINDOW = 400

_W = r"(?:premier|premi[èe]re|seconde?|deuxi[èe]me|troisi[èe]me|quatri[èe]me|cinqui[èe]me|sixi[èe]me|septi[èe]me|huiti[èe]me|neuvi[èe]me|dixi[èe]me|unique)"
RE_H_MOYEN = re.compile(rf"[Ss]ur\s+(?:le|les)\s+(?:{_W}(?:\s*,\s*|\s+et\s+|\s+))*moyens?\b|[Ss]ur\s+ce\s+moyen|[Ss]ur\s+le\s+moyen\b")
RE_MAIS_SUR = re.compile(r"\bMais,?\s+sur\s+")
RE_ENONCE_N = re.compile(r"[ÉE]nonc[ée]s?\s+du\s+moyen", re.IGNORECASE)
RE_REPONSE_N = re.compile(r"R[ée]ponse\s+de\s+la\s+Cour", re.IGNORECASE)
RE_OFFICE = re.compile(r"relev[ée]e?s?\s+d['’]office", re.IGNORECASE)
RE_SOUTIENT = re.compile(r"(?:soutient|soutiennent|conteste|contestent|fait\s+valoir|font\s+valoir|pr[ée]tend(?:ent)?|invoque(?:nt)?|selon\s+(?:la|le)\s+d[ée]fend)", re.IGNORECASE)


def _coupe(t):
    m = RE_CUT.search(t or "")
    return ((t or "")[:m.start()] if m else (t or "")).rstrip()


def _ouverture_rejet(t):
    m = RE_REJET_OUVERTURE.search(t[:OUV_WINDOW])
    if not m:
        return False
    return not RE_SOUTIENT.search(t[max(0, m.start() - 70):m.start()])


def multi_moyens(t):
    tc = _coupe(t)
    return (len(RE_H_MOYEN.findall(tc)) >= 2 or len(RE_ENONCE_N.findall(tc)) >= 2
            or len(RE_REPONSE_N.findall(tc)) >= 2 or bool(RE_MAIS_SUR.search(tc))
            or bool(RE_OFFICE.search(tc))
            or (_ouverture_rejet(tc) and bool(RE_ACCUEIL_FIN.search(tc[-FIN_WINDOW:]))))


def politique_conflits(motivation_text):
    """'accepte' | 'rejete' | 'conflit' (multi-ground or no formula found: not forced)."""
    tc = _coupe(motivation_text)
    if multi_moyens(motivation_text):
        return "conflit"
    if _ouverture_rejet(tc):
        return "rejete"
    return "accepte" if RE_ACCUEIL_FIN.search(tc[-FIN_WINDOW:]) else "conflit"


def _resoudre_conflit_politique(motiv, flags, r_codes, fam):
    """Final arbitration stage for a conflict: applies the scope-cut policy.

    accepte -> keep the cascade family, r_codes moved to trace.
    rejete  -> keep r_codes, family cleared (false positive of the cascade).
    conflit -> left unresolved.
    """
    verdict = politique_conflits(motiv)
    if verdict == 'accepte':
        return 'accepte', flags + ['c_conflit_politique_cut_accepte'], set(), fam, set(r_codes)
    if verdict == 'rejete':
        return 'rejete', flags + ['c_conflit_politique_cut_rejete'], r_codes, None, set()
    return 'conflit', flags + ['conflit_non_resolu'], r_codes, fam, set()


def derive_statut(pair, fine, fam, r_codes):
    """Derive the pair's final status.

    Blocks (pairs with no moyen_text, i.e. joint or preamble segments) go
    through the exact same checklist as every other pair: the cascade, the
    grid, then conflict or decision-outcome arbitration. This gives R1 on
    genuine manifestly-unfounded grounds, fine r-codes (R3, R5, R6, R8, R2)
    on genuine rejections, and nothing on pure joints or preambles (which
    fall back to aucun_match).
    """
    flags = []
    motiv = pair.get('motivation_text', '') or ''
    pvc_in_pair = bool(pair.get('flag_pvc', False))

    # C1, pair-level PVC: a cascade opening ground present => genuine cassation.
    if pvc_in_pair:
        if fam is not None:
            return 'accepte', flags + ['c1_pvc_pair_reclass_accepte'], r_codes, fam, set()
        return 'accepte_par_voie_de_consequence', flags + ['c1_pvc_pair'], r_codes, fam, set()

    has_axe1 = fam is not None
    has_rcodes = len(r_codes) > 0

    # C2, text-level PVC (ex-art. 624) on pairs with neither cascade nor r-codes
    if not has_axe1 and not has_rcodes:
        if RE_PVC_SHORT.search(motiv) and RE_ART_624.search(motiv):
            return 'accepte_par_voie_de_consequence', flags + ['c2_pvc_624'], set(), None, set()
        if RE_PVC_SHORT.search(motiv) and _entraine_pres(motiv):
            return 'accepte_par_voie_de_consequence', flags + ['c2_pvc_entraine'], set(), None, set()
        return 'aucun_match', flags, r_codes, None, set()

    # C2bis, cascade + isolated R1 + PVC text: PVC only if no standalone acceptance present
    if has_axe1 and RE_PVC_SHORT.search(motiv) and r_codes.issubset({'R1'}):
        if _pvc_est_principal(motiv):
            return 'accepte_par_voie_de_consequence', flags + ['c2_pvc_axe1_r1'], set(), fam, set()
        return 'accepte', flags + ['c2_pvc_axe1_r1_reclass_accepte'], set(r_codes), fam, set()

    # C4, standard cases (the rule named C3 runs later, in the post-processing below)
    if has_axe1 and not has_rcodes:
        return 'accepte', flags, r_codes, fam, set()
    if has_rcodes and not has_axe1:
        return 'rejete', flags, r_codes, None, set()

    # Conflict between a cascade family and r-codes: positional arbitration
    pos = position_arbitrage(motiv, r_codes, bloc=not (pair.get('moyen_text') or '').strip())
    if pos == 'rejete':
        return 'rejete', flags + ['position_arbitrage:rejete'], r_codes, None, set()
    if pos == 'accepte':
        return 'accepte', flags + ['position_arbitrage:accepte'], r_codes - {'R3'}, fam, set()
    # Arbitration of remaining conflicts by the decision's own outcome
    solution = pair.get('solution')
    cassation_type = pair.get('cassation_type')
    if solution == 'rejet':
        # a rejection decision -> every ground was rejected -> family cleared (false positive)
        return ('rejete',
                flags + ['c_conflit_resolu_rejet_solution'],
                r_codes, None, set())
    if solution == 'cassation' and cassation_type == 'totale':
        # No auto-resolution of full cassations by decision shape alone: the
        # scope-cut policy below decides, or honestly leaves the pair in conflict.
        return _resoudre_conflit_politique(
            motiv, flags + ['c_conflit_totale_deforce'], r_codes, fam)
    # composite-block tiebreaker: manifestly-unfounded on one ground + genuine
    # acceptance on another (same multi-ground block)
    if r_codes == {'R1'} and _composite_cassation(motiv):
        return 'accepte', flags + ['c_conflit_r1_composite'], set(r_codes), fam, set()
    # partial or ambiguous cassation: scope-cut policy, else conflict
    return _resoudre_conflit_politique(motiv, flags, r_codes, fam)


# Post-processing: rules for how r-code families interact after the scan


def _apply_C3(r_codes):
    """R8b implies R3 (co-marking post-process)."""
    if 'R8b' in r_codes and 'R3' not in r_codes:
        return r_codes | {'R3'}
    return r_codes

def _apply_C5_gating(r_codes, r_hits, fam_cascade):
    """Strip R3 when it comes only from the sovereign-assessment closing
    pattern and the cascade already found a family upstream (a likely false
    positive that would otherwise wrongly send position_arbitrage towards
    accepte).

    If fam_cascade is None: no gating (R3 stands).
    If another R3 pattern matched too: no gating (R3 has another source).
    """
    if fam_cascade is None:
        return r_codes
    if 'R3' not in r_codes:
        return r_codes
    hit_names = {h['pattern'] for h in r_hits}
    if 'R3_v48_souverainete' not in hit_names:
        return r_codes
    # Check whether another r-code-R3 pattern, besides the sovereign-assessment
    # one, also matched.
    for name in hit_names:
        if name == 'R3_v48_souverainete':
            continue
        if PATTERN_TO_FAM.get(name) == 'R3':
            return r_codes  # another R3 present, R3 stands
    # R3 comes only from the sovereign-assessment pattern, and the cascade
    # detected a family: strip R3.
    return r_codes - {'R3'}

_NSAM_RX = re.compile(
    r"(n['’]y\s+a\s+(?:donc\s+)?pas\s+lieu\s+de\s+statuer\s+par\s+une\s+d[ée]cision\s+sp[ée]cialement\s+motiv[ée]e|"
    r"article\s+1014,?\s+alin[ée]a\s+2|"
    r"alin[ée]a\s+2\s+de\s+l['’]article\s+1014|"
    r"(?:n['’]est|ne\s+sont)\s+manifestement\s+pas\s+de\s+nature\s+[àa]\s+entra[îi]ner\s+la\s+cassation)",
    re.IGNORECASE | re.UNICODE,
)
_R3_CLOSURE_RX = re.compile(
    r"(l[ée]galement\s+justifi[ée]|"
    r"(?:moyens?|griefs?)[^.;]{0,200}?(?:n['’]\s*(?:est|ont)|ne\s+(?:sont|peut|peuvent|saurait|sauraient))\s+(?:donc\s+|d[èeé]s\s*lors\s+|pas\s+)?(?:[eèéê]tre\s+)?(?:fond[ée]|accueilli|prosp[ée]rer)|"
    r"motifs?\s+exempts?\s+d['’]erreur\s+manifeste|"
    r"sans\s+encourir\s+(?:le\s+|les\s+)?griefs?|"
    r"d['’]o[uù]\s+il\s+suit\s+que\s+(?:le\s+|les\s+)?(?:moyens?|griefs?)|"
    r"(?:moyens?|griefs?)\s+(?:n['’]\s*encourt|n['’]\s*encourent)\s+pas\s+(?:de\s+)?(?:griefs?|critiques?))",
    re.IGNORECASE | re.UNICODE,
)
_ORDINAL_TOKENS = {
    'premier': 1, 'première': 1, 'premiere': 1,
    'second': 2, 'seconde': 2,
    'deuxième': 2, 'deuxieme': 2,
    'troisième': 3, 'troisieme': 3,
    'quatrième': 4, 'quatrieme': 4,
    'cinquième': 5, 'cinquieme': 5,
    'sixième': 6, 'sixieme': 6,
    'septième': 7, 'septieme': 7,
    'huitième': 8, 'huitieme': 8,
    'neuvième': 9, 'neuvieme': 9,
    'dixième': 10, 'dixieme': 10,
}
_CARDINAL_MAP = {'deux': 2, 'trois': 3, 'quatre': 4, 'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10}
_PREMIERS_RX = re.compile(r'\b(deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+premiers?\s+moyens?\b', re.IGNORECASE)
_LES_N_MOYENS_RX = re.compile(r'\bles\s+(deux|trois|quatre|cinq|six|sept|huit|neuf|dix)\s+moyens?\b', re.IGNORECASE)
_NUMERIC_ORD_RX = re.compile(r'(\d+)\s*(?:er|re|ère|ere|e|ème|eme|°)\b', re.IGNORECASE)

def _parse_nsam_targets(text, nsam_span):
    s, e = nsam_span
    win = text[max(0, s-400):min(len(text), e+400)]
    win_low = win.lower()
    targets = set()
    found = False
    for m in _PREMIERS_RX.finditer(win):
        x = _CARDINAL_MAP[m.group(1).lower()]
        for k in range(1, x+1): targets.add(k)
        found = True
    for m in _LES_N_MOYENS_RX.finditer(win):
        n = _CARDINAL_MAP[m.group(1).lower()]
        for k in range(1, n+1): targets.add(k)
        found = True
    for m in _NUMERIC_ORD_RX.finditer(win):
        n = int(m.group(1))
        if n > 30: continue
        if re.match(r"\s*(?:janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[ûu]t|septembre|octobre|novembre|d[ée]cembre)\b", win[m.end():m.end()+20].lower()): continue
        pre = win[max(0, m.start()-40):m.start()].lower()
        if 'branche' in pre[-12:]: continue
        post = win[m.end():m.end()+60].lower()
        if 'branche' in post[:30]: continue
        if re.search(r'\bmoyens?\b', post):
            targets.add(n); found = True
        elif re.search(r'^\s*(?:à|a|et)\s+\d', post):
            targets.add(n); found = True
    for word, num in _ORDINAL_TOKENS.items():
        for m in re.finditer(r'\b' + re.escape(word) + r'\b', win, re.IGNORECASE):
            pre = win[max(0, m.start()-40):m.start()].lower()
            if 'branche' in pre[-12:]: continue
            post = win[m.end():m.end()+60].lower()
            if 'branche' in post[:30]: continue
            if re.search(r'\bmoyens?\b', post):
                targets.add(num); found = True
            elif re.search(r'^\s*(?:à|a|et)\s+\w+e\s+moyens?', post):
                targets.add(num); found = True
    if re.search(r'moyens?\s+uniques?', win_low):
        targets.add(1); found = True
    autre = bool(re.search(r'(?:autres?|derniers?|tous\s+les\s+autres|seconds?)\s+moyens', win_low))
    branche_only = bool(re.search(r'\bbranches?\b', win_low)) and not found
    return targets, autre, branche_only

def _apply_collision_R1R3(r_codes, motiv, pair):
    """Collision rule for a pair carrying both R1 and R3.

    Five branches, keyed to which ground the "non spécialement motivée"
    (NSAM, art. 1014) formula actually targets in a multi-ground block: SOLO,
    INTRA and INDETERMINE keep both r-codes as-is. CROSS_MOYEN (the formula
    targets a different ground) drops R1. R3_FP (the formula is present as
    plain text but with no matching R3 closing formula nearby) drops R3.
    """
    if not ('R1' in r_codes and 'R3' in r_codes):
        return r_codes
    j_1 = (pair.get('moyen_idx') or 0) + 1
    total = pair.get('n_moyens_arret') or 1
    if total <= 1:
        return r_codes  # SOLO
    nsam_match = _NSAM_RX.search(motiv)
    has_r3_close = bool(_R3_CLOSURE_RX.search(motiv))
    if nsam_match and not has_r3_close:
        return r_codes - {'R3'}  # R3_FP
    if not nsam_match:
        return r_codes  # R1 flag-only, or INDETERMINE defaulting to INTRA
    targets, autre, branche_only = _parse_nsam_targets(motiv, nsam_match.span())
    if branche_only or (j_1 in targets):
        return r_codes  # INTRA
    if targets or autre:
        return r_codes - {'R1'}  # CROSS_MOYEN
    return r_codes  # INDETERMINE defaulting to INTRA

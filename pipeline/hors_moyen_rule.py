#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Decides whether a pair falls outside the ground's merits (hors_moyen).

A (ground, reasons) pair is hors_moyen when the reasons text is not really
answering the ground on its merits. This happens in two situations: the
reasons open with a "Portée et conséquences de la cassation" appendix
(governed by articles 1015, L.411-3 and 627), or the ground was left
unexamined because another ground already made it moot. Imported by
pipeline/classify_lib.py.
"""
import re

CAP = 220

_PFX = (r'^\s*(?:r[ée]ponse\s+(?:de\s+la\s+cour|au\s+moyen)\s*)?(?:\d{1,3}\.\s*)?(?:et\s+)?'
        r'(?:mais[, ]+)?(?:vu[^\n;]{0,90};\s*)?(?:attendu[, ]*que?[, ]+)?(?:\d{1,3}\.\s*)?')

RE_PORTEE = re.compile(_PFX + r'port[ée]e\s+et\s+cons[ée]quences?\s+de\s+la\s+cassation', re.I)

RE_MOOTCORE = re.compile(_PFX + r'(?:'
    r'le\s+rejet\s+(?:du|des|[àa]\s+intervenir)[^.;]{0,90}?(?:moyens?|pourvois?|grief)'
    r'|la\s+cassation,?\s+(?:n[\x27’]?[ée]tant\s+pas\s+prononc|[àa]\s+intervenir|prononc[ée]e?|du?\s+chef|'
    r'de\s+l[\x27’]?(?:arr[êe]t|ordonnance|jugement))[^.;]{0,130}?(?:entra[îi]ne|emporte|rend|sans\s+objet|sans\s+port)'
    r'|le?s?\s+[^.;]{0,40}?(?:moyens?|pourvois?|griefs?)(?:\s+[^.;]{0,30}?)?\s+(?:[ée]tant|ayant\s+[ée]t[ée]|ayant\s+fait\s+l[\x27’]objet\s+d[\x27’]un)\s+rejet'
    r'|le\s+pourvoi(?:[^.;]|(?<=\d)\.(?=\d)){0,60}?[ée]tant\s+rejet'
    r')', re.I)  # 130-character window: covers participle forms, multi-word
    # subjects, and a petition number sitting between the trigger and the
    # mootness marker

RE_SUBSTANTIEL = re.compile(
    r"n['’]est\s+pas\s+fond[ée]|d['’]o[ùu]\s+il\s+suit|ne\s+(?:peut|saurait)\s+(?:être|etre)\s+accueilli"
    r"|manifestement\s+pas\s+de\s+nature|n['’]y\s+a\s+pas\s+lieu\s+de\s+statuer\s+par\s+une\s+d[ée]cision\s+sp[ée]cialement"
    r"|casse\s+et\s+annule|\bviol[ée]\b|violation\s+de|appr[ée]ciation\s+souveraine|sous\s+le\s+couvert"
    r"|\birrecevable\b|ne\s+tend\s+qu['’]"
    r"|la\s+cour\s+d['’]appel\s+a\s+(?:constat|relev|reten|[ée]nonc|d[ée]duit|caract[ée]ris|exact|pu\b|fait|justifi|appr[ée]ci|d[ée]cid)"
    r"|ayant\s+(?:retenu|constat[ée]|relev[ée]|[ée]nonc[ée]|d[ée]duit|caract[ée]ris)"
    r"|mais\s+sur\s+(?:ce|le|la|les)\s+(?:moyen|branche)|par\s+motif\s+adopt", re.I)


def is_hors_moyen(motivation_text, statut):
    """True if (reasons text, status) is hors_moyen: case A (the appendix
    header) matches regardless of status, or case B (status is aucun_match,
    the text opens on a moot formula, stays under 220 characters, and no
    marker of a genuine examination is present).

    A human review of 60 cases found zero false positives.
    Some false negatives are accepted as a tradeoff (long sections, or
    blocks that combine several grounds, are not covered).
    """
    if not isinstance(motivation_text, str):
        return False
    if RE_PORTEE.match(motivation_text):
        return True
    if statut != 'aucun_match':
        return False
    if len(motivation_text) >= CAP or RE_SUBSTANTIEL.search(motivation_text):
        return False
    return bool(RE_MOOTCORE.match(motivation_text))

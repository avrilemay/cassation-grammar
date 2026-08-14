"""Decides which fine-grained ground and doctrinal family an accepted ground belongs to.

A ground is one legal complaint against the appealed decision. When the Court
accepts a ground, this file reads its wording and applies a fixed sequence of
regex patterns to decide what kind of ground it is (direct violation of the
law, lack of legal basis, misrepresentation of evidence, and so on) and which
broader family it belongs to. It only reads its input text and never changes
it. Section numbers such as 72.01 refer to Boré and Boré, La cassation en
matière civile (Dalloz, 2023).
"""
import re
import pandas as pd

REGEX_FLAGS = re.IGNORECASE | re.UNICODE
MIN_MOTIVATIONS_LEN = 50
CONTEXT_LENGTH = 300

# Patterns that read the wording of the ground itself (the T_* markers): what
# legal wrong the ground says the lower court committed, phrased the way
# rulings actually write it.
PATTERNS_T_RAW: dict[str, list[str]] = {
    # Chap. 72: Direct violation of the law (Boré Sec.72.01 et seq.)
    "T_violation": [
        # Canonical formulas of direct violation
        r"a\s+viol[eé]\s+(?:le|les|l['’]a|la)\s+(?:texte|article|disposition)",
        r"en\s+violation\s+(?:de|des|du)",
        # Demonstratives and possessives: "a violé ce/cette/ledit/lesdits/celui-ci"
        r"a\s+viol[eé]\s+(?:ce|cette|ces|cet|celui-ci|celle-ci|ledit|lesdit(?:e?s)?)\s+(?:texte|article|disposition|principe|r[èe]gle|convention)",
        # "a violé le N-ième des textes/articles"
        r"a\s+viol[eé]\s+(?:le|les)\s+(?:premier|deuxi[eè]me|second|troisi[eè]me|quatri[eè]me|dernier|m[êe]me|seconds?|premiers?)\s+(?:des|du|de)\s+(?:textes?|articles?|dispositions?)",
        # Reversal of the burden of proof = error of law
        r"a\s+invers[eé]\s+la\s+charge\s+de\s+la\s+preuve",
        r"inversion\s+de\s+la\s+charge\s+de\s+la\s+preuve",
        # Forms that tolerate extra words between "violé" and the text/article
        r"viol[eé](?:s|es|ees)?\s+(?:le|les|l['’]?\s*|la)\s+\w{1,40}?\s+(?:textes?|articles?|dispositions?|principes?)\s+susvis",
        r"viol[eé](?:s|es|ees)?\s+(?:les?|l['’]?\s*)\s+(?:textes?|articles?|dispositions?|principes?)\s+susvis",
        r"viol[eé](?:s|es|ees)?\s+l['’]?\s*article\s+[A-Z]?\.?\s*\d",
        # Same idea, with a wider gap and an optional "par ..." insert
        r"viol[eé](?:s|es|ees)?\s*(?:,\s*par\s+\w+\s+\w+\s*,)?\s*(?:le|les|l['’]?\s*|la)?.{0,40}?(?:textes?|articles?|dispositions?|principes?)\s+susvis",
        # Direct misapprehension + wrong application (ref. Ancel, Bulletin d'information de la Cour de cassation no 719)
        r"(?:a|ont)\s+m[eé]connu\s+(?:le|les|la|l['’])\s*.{0,40}?(?:textes?|articles?|dispositions?|principes?)\s+(?:susvis|pr[eé]cit|vis[eé]s?\s+au\s+moyen)",
        # Misapprehension of scope or meaning
        r"(?:a|ont)\s+m[eé]connu\s+(?:la|leur|sa)\s+(?:port[eé]e|sens)",
        # Wrong / incorrect application
        r"a\s+fait\s+(?:une\s+)?fausse\s+application\s+(?:de|des|du)",
        # "a violé par fausse application" (compound)
        r"(?:a|ont)\s+viol[eé](?:s|e|es)?\s+(?:par\s+|, ?par\s+)?(?:la\s+)?fausse\s+application",
        # "d'où il suit que [...] a violé": the ruling's own concluding sentence
        r"d['’]o[uù]\s+il\s+(?:r[eé]sulte|suit)\s+(?:qu['’]?)?(?:en\s+statuant)?[^.]{0,300}?(?:a\s+viol[eé]\s+|en\s+violation)",
        r"d['’]o[uù]\s+il\s+(?:r[eé]sulte|suit)\s+qu['’]en\s+statuant\s+(?:ainsi|comme).{0,250}?viol[eé]",
        # "a violé ces textes" (very short closing)
        r"(?:cour|conseil|tribunal|juge)\s+(?:d['’]appel\s+)?a\s+viol[eé]\s+(?:ces|le|les|l['’])\s+textes?",
        r"a\s+viol[eé]\s+(?:ces|les?)\s+textes?(?:\s+susvis)?",
        # "a ajouté au texte susvisé des conditions qu'il ne comporte pas"
        r"(?:qui\s+)?a\s+ajout[eé]\s+(?:au\s+texte|aux\s+textes?|à\s+ce(?:s|tte)?\s+(?:texte|article|disposition))\s+(?:susvis[eé]?s?\s+)?(?:des|une)\s+conditions?\s+(?:qu['’]?(?:il|elle|ils)\s+ne\s+comporte(?:nt)?\s+pas|d['’]?application\s+qu['’]?(?:il|elle|ils)\s+ne\s+pr[eé]voi(?:t|ent)\s+pas)",
    ],
    # Chap. 72: "Wrong application" sub-marker (Boré Sec.72.40 et seq.)
    # Used only to choose between VIOL_MAUVAISE_LECTURE and VIOL_DIRECTE once the
    # violation family already matched. The patterns are copied from T_violation
    # on purpose: they keep triggering the VIOLATION family through T_violation,
    # and here they additionally record that the wording points to wrong
    # application rather than direct violation.
    "T_fausse_application": [
        r"a\s+fait\s+(?:une\s+)?fausse\s+application\s+(?:de|des|du)",
        r"(?:a|ont)\s+viol[eé](?:s|e|es)?\s+(?:par\s+|, ?par\s+)?(?:la\s+)?fausse\s+application",
        r"(?:qui\s+)?a\s+ajout[eé]\s+(?:au\s+texte|aux\s+textes?|à\s+ce(?:s|tte)?\s+(?:texte|article|disposition))\s+(?:susvis[eé]?s?\s+)?(?:des|une)\s+conditions?\s+(?:qu['’]?(?:il|elle|ils)\s+ne\s+comporte(?:nt)?\s+pas|d['’]?application\s+qu['’]?(?:il|elle|ils)\s+ne\s+pr[eé]voi(?:t|ent)\s+pas)",
        # Wrong scope of application and misinterpretation (Boré Sec.72.40)
        # "champ d'application ... alors que": undue extension of a text's scope.
        r"champ\s+d['’]application\s+[^.]{0,250}?\s+alors\s+que",
        # Court (CA/tribunal/labour court/judge) + "a interprété": misinterpretation
        # of an act or a text by the trial court.
        r"(?:cour\s+d['’]appel|tribunal(?:\s+(?:judiciaire|de\s+commerce|de\s+grande\s+instance|d['’]instance))?|juge(?:s)?(?:\s+(?:du\s+fond|d['’]instance|de\s+l['’]ex[eé]cution))?|conseil\s+de\s+prud['’]hommes|juridiction(?:s)?)[^.]{0,80}?(?:qui\s+)?(?:a|ont)\s+interpr[eé]t[eé]\b",
        # Court + "a confondu": confusion of concepts by the trial court.
        r"(?:cour\s+d['’]appel|tribunal(?:\s+(?:judiciaire|de\s+commerce|de\s+grande\s+instance|d['’]instance))?|juge(?:s)?(?:\s+(?:du\s+fond|d['’]instance|de\s+l['’]ex[eé]cution))?|conseil\s+de\s+prud['’]hommes|juridiction(?:s)?)[^.]{0,80}?(?:qui\s+)?(?:a|ont)\s+confondu",
        # "a méconnu la portée / le sens": copied from T_violation on purpose.
        # It still triggers the VIOLATION family through T_violation, and here
        # it additionally flags a "misreading / misinterpretation" mode
        # (Boré Sec.72.40).
        r"(?:a|ont)\s+m[eé]connu\s+(?:la|leur|sa|le)\s+(?:port[eé]e|sens)",
        # Proximity pattern "a violé ... fausse application" (Boré Sec.72.40):
        # matches forms where "fausse application" appears together with "a violé"
        # in the same sentence (<=200 chars, no full stop). Covers:
        #   1. a comma right after "violé": "a violé, par fausse application"
        #   2. longer formulas: "a violé les textes susvisés, le premier par X,
        #      le second par fausse application"
        # Kept to the same sentence: "la cour d'appel a violé X par fausse
        # application" is already, on its own, a ground for wrong application.
        r"(?:a|ont)\s+viol[eé](?:s|e|es)?\b[^.]{0,200}?\s+(?:par\s+(?:la\s+)?)?fausse\s+application",
    ],
    # Chap. 78: Lack of legal basis / "manque de base légale" (Boré Sec.78.01 et seq.)
    "T_base_legale": [
        # Canonical formulas
        r"n['’ ]?a\s+pas\s+donn[eé]\s+de\s+base\s+l[eé]gale",
        r"n['’]?\s*(?:a|ont)\s+pas\s+donn[eé]\s+(?:à\s+(?:sa|leur)\s+d[eé]cision\s+)?(?:de\s+|une\s+)?base[s]?\s+l[eé]gale[s]?",
        r"priv[eé]\s+sa\s+d[eé]cision\s+de\s+base\s+l[eé]gale",
        # "n'a pas mis la Cour de Cassation en mesure" (also matches the OCR spacing quirk "n' a pas mis")
        r"n['’ ]?a\s+pas\s+mis\s+la\s+cour\s+de\s+cassation\s+en\s+mesure",
        r"n['’\s]*a\s+pas\s+mis\s+(?:la\s+)?cour\s+de\s+cassation\s+en\s+mesure",
        # "n'a pas satisfait aux exigences du texte/article"
        r"n['’]?\s*a\s+pas\s+satisfait\s+aux\s+exigences\s+(?:du\s+texte|des\s+textes?|de\s+l['’]?\s*article|des\s+articles?)",
        # Other civil-law formulas
        r"n['’]?\s*a\s+pas\s+l[eé]galement\s+justifi[eé]\s+sa\s+d[eé]cision",
        r"n['’]?\s*a\s+pas\s+justifi[eé]\s+sa\s+d[eé]cision\s+au\s+regard",
        r"n['’]?\s*a\s+pas\s+justifi[eé]\s+sa\s+d[eé]cision",
        # Passive form: "la décision n'est pas légalement justifiée"
        r"(?:d[eé]cision|jugement|arr[êe]t)\s+(?:attaqu[eé]e?\s+)?n['’]?\s*est\s+pas\s+l[eé]galement\s+justifi[eé]e?",
        # "méconnu les exigences" = MBL
        r"(?:a|ont)\s+m[eé]connu\s+les\s+exigences\s+(?:du|des|de\s+l['’])",
        # "a méconnu les textes susvisés" (a shorter formula)
        r"a\s+m[eé]connu\s+(?:le|les)\s+texte[s]?\s+susvis",
        r"a\s+m[eé]connu\s+(?:le|les|l['’])\s*textes?\s+(?:susvis|pr[eé]cit)",
        # "n'a pas caractérisé l'existence de": insufficient characterization
        r"n['’]?\s*(?:a|ont)\s+pas\s+caract[eé]ris[eé]\s+(?:l['’]existence|la|le)\s+\w",
        # "n'a pas permis à la Cour de cassation d'exercer son contrôle"
        r"n['’ ]?a\s+pas\s+permis\s+(?:à\s+la\s+|a\s+la\s+)?cour\s+de\s+cassation\s+d['’]exercer\s+son\s+contr[ôo]le",
        # Plural and synonyms: "ne mettent/permettent pas la Cour de cassation en mesure"
        r"ne\s+(?:mettent|mette|met|permet|permettent)\s+pas\s+(?:à\s+)?la\s+cour\s+de\s+cassation\s+(?:d['’]exercer\s+son\s+contr[ôo]le|en\s+mesure\s+d['’]?exercer)",
    ],
    "T_consequences": [
        r"n['’ ]?a\s+pas\s+tir[eé]\s+(?:toutes\s+)?les\s+cons[eé]quences\s+l[eé]gales",
        r"sans\s+tirer\s+les\s+cons[eé]quences\s+l[eé]gales",
    ],
    # Chap. 77: Lack of grounds / "défaut de motifs" (Boré Sec.77.21-23)
    # Selective on active cassation-triggering formulas only (Sec.77.21).
    "T_defaut_motifs": [
        r"ne\s+comporte\s+aucun\s+motif",
        r"sans\s+(?:[eé]noncer|exposer|donner)\s+(?:de\s+)?motifs?",
        r"sans\s+motiver\s+sa\s+d[eé]cision",
        # "n'a pas donné de motif à sa décision"
        r"n['’]?\s*(?:a|ont)\s+pas\s+donn[eé]\s+de\s+motifs?\s+(?:à\s+|a\s+)?(?:sa|leur|la)\s+d[eé]cision",
        # "n'a exprimé aucun motif"
        r"n['’]?\s*(?:a|ont)\s+exprim[eé]\s+aucun\s+motif",
        # "sans motiver leur/sa décision"
        r"sans\s+motiver\s+(?:leur|sa|la)\s+d[eé]cision",
        # Canonical visa formula (Sec.77.21)
        r"jugements?\s+doivent\s+[êe]tre\s+motiv[eé]s",
        # Cassation-triggering equivalences (Sec.77.22-23)
        r"[eé]quiva(?:lait|laient|ut|lent|le|lant)\s+(?:à\s+)?(?:l['’]?\s*)?(?:une\s+)?absence\s+de\s+motifs",
        # "sans exposer les prétentions et moyens" (visa CPC art. 455)
        r"sans\s+(?:exposer|rappeler).{0,40}les\s+pr[eé]tentions?\s+(?:et|ni|/)?\s*(?:les\s+)?moyens?",
    ],
    "T_motifs_contradictoires": [
        r"motifs\s+contradictoires",
        r"contradiction\s+de\s+motifs",
    ],
    # Chap. 77: Failure to respond to submissions (Sec.77.24)
    # Cassation-triggering formulas + visa art. 455 CPC within a 500-char window.
    "T_defaut_reponse": [
        # Canonical formulas
        r"n['’ ]?a\s+pas\s+r[eé]pondu\s+aux\s+conclusions",
        r"sans\s+r[eé]pondre\s+aux\s+conclusions",
        # Other prepositions, and "moyen(s)" instead of "conclusions"
        # Also matches the OCR spelling "a ces" (missing the accent on "à")
        r"sans\s+r[eé]pondre\s+(?:[àa]\s+ces|aux?)\s+conclusions",
        r"sans\s+r[eé]pondre\s+au\s+moyen",
        r"sans\s+r[eé]pondre\s+aux\s+moyens",
        r"d[eé]faut\s+de\s+r[eé]ponse\s+(?:à|aux)\s+conclusions",
        r"d[eé]faut\s+de\s+r[eé]ponse\s+à\s+conclusions\s+constitue\s+un\s+d[eé]faut\s+de\s+motifs",
        # Combination of visa art. 455 CPC + cassation-triggering formula (500-char window)
        r"vu\s+l['’]article\s+455\s+(?:al(?:in[eé]a)?\.?\s*\d\s+)?du\s+(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile[^.]{0,500}?(?:r[eé]pondu|r[eé]pondre|sans\s+r[eé]pondre)",
    ],
    # Chap. 79: Misrepresentation of evidence / "dénaturation" (Boré Sec.79.01 et seq.)
    "T_denaturation": [
        r"a\s+d[eé]natur[eé]",
        r"(?<!sans\s)(?<!ni\s)d[eé]naturation\s+(?:du|des|de\s+la)",  # negation guard: excludes "sans dénaturation"
        r"en\s+d[eé]naturant",
        r"(?:en\s+)?ont\s+d[eé]natur[eé]\s+(?:les?\s+termes?)?",
        r"les\s+juges?\s+(?:du\s+fond\s+)?(?:en\s+)?ont\s+d[eé]natur[eé]",
        # "sans dénaturer" is a negation that reverses the verdict: "la cour, sans
        # dénaturer X, a retenu Y" means the trial court did not misrepresent the
        # evidence, so this pattern must not trigger DENATURATION.
        # Canonical procedural visa for misrepresentation of evidence (keep)
        r"vu\s+l['’]obligation\s+(?:pour\s+le\s+juge\s+)?de\s+ne\s+pas\s+d[eé]naturer",
    ],
    "T_motifs_inoperants": [
        r"par\s+(?:un|des)\s+motifs?\s+inop[eé]rants?",
        r"motif\s+inop[eé]rant",
    ],
    "T_motifs_impropres": [
        r"par\s+(?:un|des)\s+motifs?\s+impropres?",
        r"motif\s+impropre\s+[aà]\s+caract[eé]riser",
    ],
    "T_motifs_dubitatifs": [
        r"motifs\s+dubitatifs",
        r"motifs?\s+hypoth[eé]tiques?",
        r"motifs?\s+inintelligibles?",
        r"motifs?\s+ambigus?",
        # "motifs d'ordre général" / "motifs généraux" intentionally excluded: those
        # in fact denote insufficient characterization (-> MOTIFS_IMPROPRES/MBL),
        # not a dubitative ground.
    ],
    # Chap. 74 sect. 2: Adversarial principle / "contradictoire" (Sec.74.21 et seq.)
    # Extended to the rights of the defence (art. 14/15 CPC) + ECHR art. 6-1.
    "T_proc_contradictoire": [
        r"principe\s+de\s+la\s+contradiction",
        r"article\s+16\s+du\s+code\s+de\s+proc[eé]dure\s+civile",
        r"m[eé]connu\s+(?:le|son)\s+(?:principe\s+du\s+)?contradictoire",
        # Rights of the defence (art. 14/15 CPC), singular and plural, "nouveau code"
        r"droits?\s+de\s+la\s+d[eé]fense",
        r"article\s+(?:14|15)\s+du\s+code\s+de\s+proc[eé]dure\s+civile",
        r"articles?\s+(?:14|15|16)\s+(?:al(?:in[eé]a)?\.?\s*\d\s+)?du\s+(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile",
        # Ground raised sua sponte without invitation to submit observations (art. 16 al. 3 CPC)
        r"n['’]?\s*(?:a|ont)\s+pas\s+invit[eé]\s+les?\s+parties?\s+(?:à\s+pr[eé]senter|a\s+pr[eé]senter)\s+leurs?\s+observations",
        # ECHR art. 6-1: independent and impartial tribunal
        r"tribunal\s+(?:n['’]?est\s+pas\s+|n['’]a\s+pas\s+[eé]t[eé]\s+)?ind[eé]pendant\s+et\s+impartial",
    ],
    # Chap. 73 sect. 2: Ultra vires / "excès de pouvoir" (Boré Sec.73.21 et seq.)
    # Lexical formulas + procedural visas art. 4/5/12 NCPC (the code de
    # procédure civile under its former name).
    "T_exces_pouvoir": [
        # Markers about the scope of the dispute (art. 4, and the devolutive
        # effect of appeal, art. 562, Bore 122.45), not about a ruling quashed
        # only as a consequence of another ground (PVC, art. 624 CPC): these
        # patterns must never match on article 624.
        r"(?:a\s+)?m[eé]connu\s+l['’]\s*objet\s+du\s+litige",
        r"(?:a\s+)?d[eé]natur[eé]\s+(?:l['’]\s*objet|les?\s+termes?)\s+du\s+litige",
        r"aggraver\s+le\s+sort\s+de\s+l['’]appel",
        r"ne\s+statue\s+que\s+sur\s+les\s+pr[eé]tentions",
        r"(?:a\s+)?m[eé]connu\s+(?:ses\s+pouvoirs|l['’][eé]tendue\s+de\s+ses\s+pouvoirs)",
        r"statu[eé]\s+(?:ultra|extra)\s+petita",
        # Cassation-triggering formulas: disregard of the terms of the dispute
        r"a\s+modifi[eé]\s+l['’]?\s*objet\s+du\s+litige",
        r"a\s+m[eé]connu\s+les?\s+termes?\s+du\s+litige",
        # Disregard of the court's own remit
        r"a\s+m[eé]connu\s+(?:son|les?\s+limites?\s+de\s+son|l['’]?[eé]tendue\s+de\s+son)\s+office",
        r"m[eé]connaissance\s+de\s+(?:son|l['’]?[eé]tendue\s+de\s+son)\s+office",
        # Other formulas for the same idea
        r"a\s+modifi[eé]\s+les?\s+termes?\s+du\s+litige",
        r"a\s+outrepass[eé]\s+(?:ses|son)\s+pouvoirs?",
        r"a\s+exc[eé]d[eé]\s+(?:ses|son)\s+pouvoirs?",
        # Procedural visas art. 4/5/12 NCPC + cassation-triggering formula (400-char window)
        r"vu\s+l['’]article\s+4\s+du\s+(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile[^.]{0,400}?(?:modifi[eé]|m[eé]connu|outrepass|a\s+viol[eé])",
        r"a\s+(?:refus[eé]|omis)\s+(?:de\s+|d['’])?[eé]valuer\s+(?:la|le|les)",
    ],
    # Chap. 76: Loss of legal basis / "perte de fondement juridique" (Boré Sec.76.01 et seq.)
    # Canonical formulas only. "Par voie de conséquence" (PVC, art. 624 CPC,
    # Bore 122.45) is excluded on purpose: that is the automatic effect of
    # another ground being upheld, not an independent ground on its own.
    "T_perte_fondement": [
        r"(?:d[eé]cision|arr[êe]t)\s+(?:se\s+trouve\s+)?priv[eé]e?\s+de\s+fondement\s+juridique",
        r"perte?\s+de\s+fondement\s+juridique",
        r"perd(?:re|u|ant)?\s+(?:son|leur)\s+fondement\s+juridique",
        r"(?:est|sont|se\s+trouve(?:nt)?)\s+priv[eé]e?s?\s+de\s+(?:tout\s+)?fondement\s+juridique",
    ],
    "T_motifs_disp_contradiction": [
        # Selective on the cassation-triggering context (Boré Sec.77.22.B)
        r"contradiction\s+entre\s+(?:les?\s+)?motifs?\s+et\s+(?:le\s+)?dispositif\s+[eé]quivaut\s+(?:à|au)\s+d[eé]faut\s+de\s+motifs",
        r"enta[cs]h[eé]e?\s+d['’]?\s*une\s+contradiction\s+entre\s+(?:les?\s+)?motifs?\s+et\s+(?:le\s+)?dispositif",
        # Other formulas for the same motifs/dispositif contradiction (Bore
        # Sec.77.22.B): the plain wording "contradiction entre les motifs et le
        # dispositif", and rearranged forms such as "retenu au dispositif ...
        # écartait dans les motifs" or "a relevé dans ses motifs ... dans le
        # dispositif".
        r"contradiction\s+entre\s+(?:les?\s+)?motifs?\s+et\s+(?:le\s+)?dispositif",
        r"a\s+relev[eé]\s+(?:dans\s+ses\s+motifs|au\s+motif).{0,200}?(?:dans\s+(?:son|le)\s+dispositif|au\s+dispositif)[^.]{0,40}?\s+(?:a\s+|ordonn|d[eé]cid|condamn)",
        r"(?:retenu|retenant)\s+au\s+dispositif.{0,200}?[eé]cart(?:e|ait|ant)\s+(?:dans\s+les\s+motifs|au\s+motif)",
    ],
    # Chap. 71: Failure to rule / ultra petita (Boré Sec.71.02, Sec.71.09, Sec.71.12)
    "T_omission_ultra_petita": [
        # Visa art. 463 NCPC (rectification procedure) intentionally excluded: a bare
        # visa on 463 matches the topic, not the actual verdict.
        # Awarded more than / different from what was claimed
        r"accord[eé]\s+(?:plus|davantage)\s+qu['’]?(?:il|on)\s+n['’]?(?:a|avait)\s+[eé]t[eé]\s+demand",
    ],
    # Chap. 73 sect. 1: Lack of jurisdiction / "incompétence" (Boré Sec.73.11-13)
    # Procedural visa NCPC 76-77 or art. 92 (raised sua sponte, order of jurisdiction)
    # or art. 4 Civil Code combined with a cassation-triggering formula (denial of justice).
    "T_incompetence": [
        # "raising the lack of jurisdiction sua sponte"
        r"relev(?:e|er|ant|é)\s+d['’]?\s*office\s+(?:le\s+moyen\s+tir[eé]\s+de\s+)?l['’]incomp[eé]tence",
    ],
    # Chap. 74 sect. 1: Defect of form / "vice de forme" (Boré Sec.74.11-82)
    # NCPC visa for composition/signature (447, 449, 454, 456, 457, 458, 749),
    # strict whitelist excluding article 455 (grounds of judgment -> DEFAUT_MOTIFS/MBL).
    "T_vice_forme": [
        r"vu\s+l['’]article\s+456\s+du\s+(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile",
        # Composite visa (whitelist 447/449/454/456/457/458/749, excluding 455)
        r"(?:vu|ensemble|par\s+application\s+de)\s+(?:l['’]article|les?\s+articles?)\s+(?:(?:447|449|454|456|457|458|749)[\s,]+(?:et\s+)?)*(?:456|458)(?:[\s,]+(?:et\s+)?(?:447|449|454|456|457|458|749))*\s+(?:du\s+|de\s+l['’]?\s*)(?:nouveau\s+)?code\s+de\s+proc[eé]dure\s+civile",
        # "il ne résulte pas des mentions du jugement"
        r"(?:il\s+)?ne\s+r[eé]sulte\s+pas\s+des\s+mentions?\s+(?:du\s+)?jugements?\s+que",
        # Composition / odd number of judges
        r"arr[êe]ts\s+des\s+cours\s+d['’]appel\s+sont\s+rendus\s+par\s+trois\s+magistrats",
        r"d[eé]lib[eé]rant\s+en\s+nombre\s+impair",
        r"m[eé]connaissance\s+de\s+la\s+r[èe]gle\s+de\s+l['’]imparit[eé]",
        r"inobservation\s+de\s+l['’]imparit[eé]",
        # "audience solennelle" (L. 212-2 COJ) left out on purpose: it never fires here
    ],
    # Chap. 75: Conflicting judgments / "contrariété de jugements" (Boré Sec.75.02-08,
    # art. 617/618 CPC). No pattern is defined for this family here: its only
    # candidate pattern (visa art. 617/618 NCPC) never matched on the reference
    # corpus, so the key is left out on purpose. The cascade below reads this
    # flag with a safe default (see `_cascade_axe1`), so a missing key behaves
    # exactly like an always-false marker.
}


# Patterns that read the kind of legal reasoning behind the ground (the J_*
# markers), not just its wording: for example, drawing the wrong conclusions
# from the lower court's own findings, or skipping a factual check the ground
# needed.
PATTERNS_J_RAW: dict[str, list[str]] = {
    # Chap. 72 Sec.66.22: "Résistance" (drawing the wrong legal conclusions from
    # the trial court's own findings).
    "J_resistance": [
        r"alors\s+qu['’]il\s+r[eé]sultait\s+de\s+(?:ses|ces)\s+(?:propres\s+)?constatations",
        # Present tense, in addition to the imperfect
        r"alors\s+qu['’]?\s*il\s+r[eé]sulte\s+de\s+(?:ses|ces)\s+(?:propres\s+)?constatations",
        r"alors\s+qu['’]elle\s+(?:avait\s+)?(?:constat|relev|retenu)",
        r"alors\s+qu['’]il\s+(?:avait\s+)?(?:constat|relev|retenu)",
        r"de\s+ses\s+propres\s+constatations",
        r"il\s+r[eé]sultait\s+des?\s+(?:propres\s+)?constatations\s+(?:de\s+la\s+cour|du\s+juge)",
        # "n'a pas tiré les conséquences légales de ses [propres] constatations"
        # (tolerates the OCR typo "contatations")
        r"n['’]?\s*a\s+pas\s+tir[eé]\s+les\s+cons[eé]quences\s+l[eé]gales\s+de\s+ses\s+(?:propres\s+)?cons?tatations",
        r"n['’]?\s*a\s+pas\s+tir[eé]\s+(?:les?\s+|toutes\s+les\s+)?cons[eé]quences\s+(?:l[eé]gales\s+)?de\s+ses\s+(?:propres\s+)?constatations",
        # "n'a pas tiré les conséquences (légales) de ses (propres) énonciations"
        r"n['’]?\s*(?:a|ont)\s+pas\s+tir[eé]\s+(?:les?\s+)?cons[eé]quences\s+(?:l[eé]gales\s+)?de\s+ses\s+(?:propres\s+)?[eé]nonciations",
    ],
    # Chap. 72 Sec.72.06: Explicit refusal to apply the law
    "J_refus_explicite": [
        r"par\s+refus\s+d['’]?\s*application",
        r"viol[eé](?:s|es|ees)?\s+(?:le|les|l['’]?\s*|la)?\s*(?:texte|article|disposition|principe).{0,120}?\s+par\s+refus\s+d['’]?\s*application",
    ],
    # Chap. 78 Sec.78.05: Omitted factual investigation
    # All verb forms: infinitive + past participle + negations.
    "J_investigation_omise": [
        r"sans\s+(?:avoir\s+)?recherch[eé](?:r)?\b",
        r"sans\s+(?:avoir\s+)?constat[eé](?:r)?\b",
        r"sans\s+(?:avoir\s+)?relev[eé](?:r)?\b",
        r"sans\s+(?:avoir\s+)?caract[eé]ris[eé](?:r)?\b",
        r"sans\s+s['’]?\s*expliquer\s+sur",
        r"sans\s+(?:avoir\s+)?pr[eé]cis[eé](?:r)?\b",
        # "s'est bornée à" intentionally excluded: too generic a pattern
        # (MBL_PUR != omitted investigation).
        r"sans\s+(?:appr[eé]cier|examiner)\s+(?:concr[eè]tement|si|la)",
        # A few more forms
        r"sans\s+v[eé]rifier",
        r"sans\s+s['’]assurer\s+(?:de|si|que)",
        # Equivalent negative forms
        r"n['’]?\s*a\s+pas\s+recherch[eé]\b",
        r"n['’]?\s*a\s+pas\s+(?:v[eé]rifi[eé]|caract[eé]ris[eé]|examin[eé])\b",
        # "faute pour la CA d'avoir"
        r"faute\s+(?:pour\s+(?:la\s+)?(?:cour|conseil|tribunal)\s+(?:d['’]appel\s+)?)?d['’]?avoir\s+(?:recherch|v[eé]rifi|caract[eé]ris|constat|examin)",
        # "devait/aurait dû"
        r"(?:devait|aurait\s+d[uû])\s+(?:rechercher|v[eé]rifier|caract[eé]riser|constater)\s+si",
        # "alors qu'il/elle/ils n'avait/n'avaient pas constaté/relevé/recherché/..."
        r"alors\s+qu['’](?:il|elle|ils)\s+n['’]?\s*avai(?:t|en)t\s+pas\s+(?:constat[eé]|relev[eé]|recherch[eé]|v[eé]rifi[eé]|examin[eé]|pr[eé]cis[eé]|caract[eé]ris[eé])",
        # "sans expliquer en quoi"
        r"sans\s+expliquer\s+en\s+quoi",
        # "sans relever aucune (autre) circonstance/élément"
        r"sans\s+relever\s+aucune?\s+(?:autre\s+)?(?:circonstance|[eé]l[eé]ment)",
        # "sans préciser ni analyser"
        r"sans\s+pr[eé]ciser\s+ni\s+analyser",
        # "sans fournir aucun élément"
        r"sans\s+fournir\s+aucun\s+[eé]l[eé]ment",
        # Cassation-triggering formula "il (lui) appartenait de + investigation verb"
        # (rechercher, vérifier, caractériser, constater, examiner, apprécier,
        # déterminer, procéder, appréhender). Doctrinally = omitted factual
        # investigation (Boré Sec.78.05). The verbs "statuer / se prononcer /
        # répondre" are intentionally excluded (they belong to the court's remit,
        # not to fact-finding).
        r"il\s+(?:lui\s+)?appartenait\s+[^.]{0,60}?(?:d['’]|de\s+)(?:rechercher|v[eé]rifier|caract[eé]riser|constater|examiner|appr[eé]cier|d[eé]terminer|proc[eé]der|appr[eé]hender)",
    ],
    # This table has no J_scaffold_alors_que key. Such a flag would mark the
    # rhetorical scaffold "en statuant ainsi, alors que..." in its canonical,
    # pronominal, and multi-branch forms. Nothing downstream reads a flag
    # like that, so leaving it out changes no output. The wording alone is
    # common, but it never decides a classification by itself.
}


# Presence/absence patterns (the M_* markers): a plain yes/no flag for wording
# such as an omitted investigation. The cascade above does not read these flags.
# The tables are consumed by `extract_m_contextes` below and by consumers
# outside this repository (the consultation app, article-level analyses). Do
# not remove.
PATTERNS_M_RAW: dict[str, list[str]] = {
    "M_sans_rechercher": [
        r"sans\s+(?:avoir\s+)?recherch[eé]",
        r"n['’]?\s*a\s+pas\s+recherch[eé]\b",
    ],
    "M_sans_constater":    [r"sans\s+(?:avoir\s+)?constat[eé]"],
    "M_sans_relever":      [r"sans\s+(?:avoir\s+)?relev[eé]"],
    "M_sans_preciser":     [r"sans\s+(?:avoir\s+)?pr[eé]cis[eé]"],
    "M_sans_expliquer":    [r"sans\s+s['’]expliquer"],
    "M_sans_caracteriser": [r"sans\s+caract[eé]riser"],
    "M_sest_bornee":       [r"s['’]?est\s+born[eé][es]?\s+(?:à|a)\s+"],
    # Negative lookahead to avoid double-counting with J_resistance
    "M_alors_que": [r"alors\s+que\b(?!\s+ces\s+constatations)"],
}

# Typed patterns to capture the content following each investigation marker.
# Each entry is (marker_name, raw regex). The name matches exactly the boolean
# M_* flag, so that later analyses can filter by type.
#
# Table kept intentionally intact: used only by `extract_m_contextes` (text
# context after each marker, out["M_contextes"]). Neither `_cascade_axe1` nor
# hybrid_lib.py reads it. It is purely diagnostic and contextual: no
# classification branch depends on it. Do not remove without re-checking call
# sites.
#
# Types:
#   - 6 "sans X" markers: missing factual criterion
#     (usable for a substantive Dalloz-style grid).
#   - M_sest_bornee: what the trial court actually found (insufficient).
#   - M_alors_que: legal counter-argument (positive claim, not a factual gap).
PATTERNS_M_CONTEXTES_RAW: list[tuple[str, str]] = [
    ("M_sans_rechercher",   r"sans\s+(?:avoir\s+)?recherch[eé](?:r)?\b|n['’]?\s*a\s+pas\s+recherch[eé]\b"),
    ("M_sans_constater",    r"sans\s+(?:avoir\s+)?constat[eé](?:r)?\b"),
    ("M_sans_caracteriser", r"sans\s+(?:avoir\s+)?caract[eé]ris[eé](?:r)?\b"),
    ("M_sans_expliquer",    r"sans\s+s['’]?\s*[eê]tre\s+expliqu[eé]e?\s+sur|sans\s+s['’]?\s*expliquer\s+sur"),
    ("M_sans_relever",      r"sans\s+(?:avoir\s+)?relev[eé](?:r)?\b"),
    ("M_sans_preciser",     r"sans\s+(?:avoir\s+)?pr[eé]cis[eé](?:r)?\b"),
    ("M_sest_bornee",       r"s['’]?\s*est\s+born[eé]e?\s+(?:à|a)"),
    ("M_alors_que",         r"alors\s+que\b(?!\s+ces\s+constatations)"),
]


# Statutory-reference patterns (the V_* markers): whether a specific legal
# article is cited in the ruling's visa (the block of legal provisions listed
# at its head) or, failing that, in the raw text. The cascade itself does not
# read these flags. The tables are consumed by consumers outside this
# repository (the consultation app, served separately, and article-level
# analyses). Do not remove.
PATTERNS_V_RAW: dict[str, list[str]] = {
    "V_L1221_1": [r"L\.?\s*1221[-–—\s]?1\b"],
    "V_L8221_5": [r"L\.?\s*8221[-–—\s]?5\b"],
    "V_L8221_6": [r"L\.?\s*8221[-–—\s]?6\b"],
    "V_L1242_1": [r"L\.?\s*1242[-–—\s]?1\b"],
    "V_L1245_1": [r"L\.?\s*1245[-–—\s]?1\b"],
    "V_L3123_14": [r"L\.?\s*3123[-–—\s]?14\b"],
    "V_455_CPC": [r"article\s+455\s+(?:du\s+)?code\s+de\s+proc[eé]dure\s+civile"],
    "V_12_CPC":  [r"article\s+12\s+(?:du\s+)?code\s+de\s+proc[eé]dure\s+civile"],
    "V_16_CPC":  [r"article\s+16\s+(?:du\s+)?code\s+de\s+proc[eé]dure\s+civile"],
}


# Compiled regexes, ready to run against a text.
COMPILED_T = {k: [re.compile(p, REGEX_FLAGS) for p in v] for k, v in PATTERNS_T_RAW.items()}
COMPILED_J = {k: [re.compile(p, REGEX_FLAGS) for p in v] for k, v in PATTERNS_J_RAW.items()}
COMPILED_M = {k: [re.compile(p, REGEX_FLAGS) for p in v] for k, v in PATTERNS_M_RAW.items()}
COMPILED_V = {k: [re.compile(p, REGEX_FLAGS) for p in v] for k, v in PATTERNS_V_RAW.items()}
COMPILED_M_CONTEXTES = [(name, re.compile(p, REGEX_FLAGS)) for name, p in PATTERNS_M_CONTEXTES_RAW]

# From a fine-grained ground to its doctrinal family.
FAMILLE_MAPPING: dict[str, str] = {
    # Chap. 71: Failure to rule / ultra petita
    "OMISSION_ULTRA_PETITA": "OMISSION_ULTRA_PETITA",
    # Chap. 72 + Sec.66.22: Violation (4 rhetorical modalities)
    "RESISTANCE": "VIOLATION",
    "REFUS_EXPLICITE": "VIOLATION",
    "VIOL_MAUVAISE_LECTURE": "VIOLATION",
    "VIOL_DIRECTE": "VIOLATION",
    # Chap. 73 + 76: Ultra vires and loss of legal basis
    "INCOMPETENCE": "EXCES_OFFICE",
    "EXCES_POUVOIR": "EXCES_OFFICE",
    "PERTE_FONDEMENT": "EXCES_OFFICE",
    # Chap. 74: Defect of form + adversarial principle
    "VICE_FORME": "VICE_FORME",
    "VIOL_PROC_CONTRADICTOIRE": "VICE_FORME",
    # Chap. 75: Conflicting judgments
    "CONTRARIETE_JUGEMENTS": "CONTRARIETE",
    # Chap. 77: Lack of grounds (formal defects)
    "DEFAUT_MOTIFS": "VICE_MOTIFS",
    "DEFAUT_REPONSE": "VICE_MOTIFS",
    "MOTIFS_CONTRADICTOIRES": "VICE_MOTIFS",
    "MOTIFS_DISP_CONTRADICTION": "VICE_MOTIFS",
    "MOTIFS_DUBITATIFS": "VICE_MOTIFS",
    # Chap. 78: Lack of legal basis (4 sub-modalities)
    "MBL_PUR": "MBL",
    "MBL_INVESTIGATION": "MBL",
    "MOTIFS_INOPERANTS": "MBL",
    "MOTIFS_IMPROPRES": "MBL",
    # Chap. 79: Misrepresentation of evidence
    "DENATURATION": "DENATURATION",
}


def assign_famille(cas):
    """Map a fine cas_ouverture_axe1 to its doctrinal family.

    Returns a missing value when the input is empty (labelled ANGLE_MORT downstream).
    """
    if cas is None or (isinstance(cas, float) and pd.isna(cas)):
        return pd.NA
    return FAMILLE_MAPPING.get(cas, pd.NA)


# Running every pattern and walking the cascade.
def extract_m_contextes(motivations_norm: str) -> list[dict]:
    """Capture the CONTEXT_LENGTH (300) chars following each match of one of the 8
    investigation markers. Returns a list of dicts {marker, text}.

    - 6 "sans X" markers: missing factual criterion
    - M_sest_bornee: what the trial court actually did (insufficient)
    - M_alors_que: legal counter-argument (positive claim)
    """
    if not isinstance(motivations_norm, str):
        return []
    contextes: list[dict] = []
    for marker_name, pat in COMPILED_M_CONTEXTES:
        for m in pat.finditer(motivations_norm):
            end = m.end()
            text = motivations_norm[end:end + CONTEXT_LENGTH].strip()
            contextes.append({"marker": marker_name, "text": text})
    return contextes


def detect_axe1(motivations_norm: str, visa_text: str, text_full: str) -> dict:
    """Compute all T_*, J_*, M_*, V_* flags plus cas_ouverture_axe1.

    motivations_norm : lowercased grounds text, with &apos; already decoded upstream.
    visa_text        : flattened visa field (titles), or a fallback to the full text.
    text_full        : raw `text` (used for V_* visa codes when visa_text is empty).
    """
    short = (not isinstance(motivations_norm, str)) or len(motivations_norm) < MIN_MOTIVATIONS_LEN

    out: dict = {}

    # T_* / J_* / M_* booleans: False when the grounds text is too short.
    if short:
        for code in COMPILED_T:
            out[code] = False
        for code in COMPILED_J:
            out[code] = False
        for code in COMPILED_M:
            out[code] = False
        out["M_contextes"] = []
        out["n_marqueurs_invest"] = 0
    else:
        for code, pats in COMPILED_T.items():
            out[code] = any(p.search(motivations_norm) for p in pats)
        for code, pats in COMPILED_J.items():
            out[code] = any(p.search(motivations_norm) for p in pats)
        n_marqueurs = 0
        for code, pats in COMPILED_M.items():
            present = any(p.search(motivations_norm) for p in pats)
            out[code] = present
            if present:
                n_marqueurs += 1
        out["M_contextes"] = extract_m_contextes(motivations_norm)
        out["n_marqueurs_invest"] = n_marqueurs

    # V_* visa codes: haystack = concatenated visa, else the raw text.
    haystack = visa_text if visa_text else (text_full or "")
    for code, pats in COMPILED_V.items():
        out[code] = any(p.search(haystack) for p in pats)

    # cas_ouverture_axe1 cascade
    out["cas_ouverture_axe1"] = _cascade_axe1(out, short)
    return out


def _cascade_axe1(flags: dict, short: bool) -> str | None:
    """Walk the flags in a fixed order and return the first ground that matches.

    Families are tested in this order:
      1. OMISSION_ULTRA_PETITA
      2. EXCES_OFFICE  (lack of jurisdiction, loss of legal basis, ultra vires)
      3. VICE_FORME    (defect of form, adversarial principle)
      4. CONTRARIETE
      5. VICE_MOTIFS   (failure to respond, contradictory/dubitative/disp motifs, lack of grounds)
      6. DENATURATION
      7. MBL           (lack of legal basis)
      8. VIOLATION     (résistance, explicit refusal, wrong application, direct)

    Within a family, the order still follows how specific each pattern is, so
    the fine-grained ground returned stays meaningful even when several
    families could apply. Grouping the order by family first, rather than by
    Bore chapter, keeps the logic easy to read without changing the resulting
    accuracy.
    """
    if short:
        return None

    # 1. OMISSION / ULTRA PETITA
    if flags.get("T_omission_ultra_petita", False):
        return "OMISSION_ULTRA_PETITA"

    # 2. EXCES D'OFFICE
    if flags.get("T_incompetence", False):
        return "INCOMPETENCE"
    if flags["T_perte_fondement"]:
        return "PERTE_FONDEMENT"
    if flags["T_exces_pouvoir"]:
        return "EXCES_POUVOIR"

    # 3. VICE DE FORME
    if flags.get("T_vice_forme", False):
        return "VICE_FORME"
    if flags["T_proc_contradictoire"]:
        return "VIOL_PROC_CONTRADICTOIRE"

    # 4. CONTRARIETE DE JUGEMENTS
    if flags.get("T_contrariete_jugements", False):
        return "CONTRARIETE_JUGEMENTS"

    # 5. VICE DE MOTIFS (failure to respond + other defects)
    if flags["T_defaut_reponse"]:
        return "DEFAUT_REPONSE"
    if flags["T_motifs_disp_contradiction"]:
        return "MOTIFS_DISP_CONTRADICTION"
    if flags["T_motifs_contradictoires"]:
        return "MOTIFS_CONTRADICTOIRES"
    if flags["T_motifs_dubitatifs"]:
        return "MOTIFS_DUBITATIFS"
    if flags["T_defaut_motifs"]:
        return "DEFAUT_MOTIFS"

    # 6. DENATURATION
    if flags["T_denaturation"]:
        return "DENATURATION"

    # 7. MANQUE DE BASE LEGALE
    if flags["T_motifs_inoperants"]:
        return "MOTIFS_INOPERANTS"
    if flags["T_motifs_impropres"]:
        return "MOTIFS_IMPROPRES"
    if flags["J_investigation_omise"] and (flags["T_base_legale"] or flags["T_consequences"]):
        return "MBL_INVESTIGATION"
    if flags["T_base_legale"]:
        return "MBL_PUR"

    # 8. VIOLATION (last)
    if flags["J_resistance"]:
        return "RESISTANCE"
    if flags["J_refus_explicite"]:
        return "REFUS_EXPLICITE"
    if flags["T_fausse_application"]:
        return "VIOL_MAUVAISE_LECTURE"
    if flags["T_violation"]:
        return "VIOL_DIRECTE"

    return None

def classify(motivation_text: str) -> tuple[str, str]:
    """Simplified public API: classify a grounds text into (cas_ouverture_axe1, famille_axe1).

    Useful outside the pipeline, for ad hoc analysis or sanity checks.
    Returns ("ANGLE_MORT", "ANGLE_MORT") if the grounds text is empty or unrecognized.
    """
    if not isinstance(motivation_text, str) or len(motivation_text.strip()) < 1:
        return "ANGLE_MORT", "ANGLE_MORT"
    txt = motivation_text.replace("&apos;", "'").lower()
    flags = detect_axe1(txt, "", motivation_text)
    fine = flags["cas_ouverture_axe1"] or "ANGLE_MORT"
    fam_value = assign_famille(fine)
    fam = fam_value if isinstance(fam_value, str) else "ANGLE_MORT"
    return fine, fam

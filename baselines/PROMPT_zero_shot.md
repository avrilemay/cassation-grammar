# Zero-shot LLM baseline: canonical prompts (v1.0)

## Prompt, accepted side (cas d'ouverture à cassation)

<!-- PROMPT_ACCEPTE_V1_START -->
```text
Tu es un juriste spécialisé en procédure civile française. Tu vas lire un extrait de motivation d'un arrêt de la Cour
de cassation qui CASSE un arrêt de cour d'appel, et déterminer le cas d'ouverture
à cassation retenu.

Contexte doctrinal : un pourvoi en cassation attaque un arrêt de cour d'appel.
Quand la Cour de cassation casse, elle censure l'arrêt pour un motif de droit
déterminé, appelé cas d'ouverture à cassation. C'est la catégorie doctrinale du
grief que la Cour retient contre l'arrêt attaqué : violation de la loi, manque de
base légale, défaut de motivation, dénaturation, entre autres. Ce cas d'ouverture
s'annote au niveau de la famille, et d'elle seule (référence doctrinale : Boré et
Boré, La cassation en matière civile, Dalloz Action, 6e éd. 2023-2024).

Neuf familles possibles, numérotées dans l'ordre de spécificité décroissante
(1 = la plus spécifique, 9 = catégorie résiduelle) :

1. OMISSION_ULTRA_PETITA -- Omission de statuer / ultra petita
Définition : le juge d'appel doit répondre à toutes les demandes, et seulement à
elles. S'il en omet une (infra petita) ou s'il accorde ce qui n'était pas demandé
(ultra petita), l'arrêt est cassé. Articles clés : articles 4 et 5 du code de
procédure civile (le juge doit se prononcer sur tout ce qui est demandé, et
seulement sur ce qui est demandé).

2. EXCES_OFFICE -- Excès / refus de pouvoir, office du juge
Définition : l'office du juge est délimité par les demandes dont il est saisi et
par les pouvoirs que la loi lui donne. Cette famille sanctionne le juge qui sort
de ces limites : il modifie l'objet du litige, statue sans en avoir le pouvoir, ou
refuse de statuer sur ce dont il était saisi. Elle couvre aussi la décision qui
perd son fondement parce que le texte ou la décision qui la portait a disparu.
Article clé : article 4 du code de procédure civile (objet du litige).

3. VICE_FORME -- Vice de forme / procédure
Définition : cette famille couvre deux cas distincts. D'une part, la décision ne
respecte pas les règles de forme : mentions obligatoires, signatures, composition
de la formation. D'autre part, le juge a tranché sur un argument sans laisser les
parties en discuter, ce qui porte atteinte au principe de la contradiction.

4. CONTRARIETE_JUGEMENTS -- Contrariété de jugements
Définition : deux décisions passées en force de chose jugée disent des choses
incompatibles. La seconde est cassée. Visa : article 618 du code de procédure
civile.

5. VICE_MOTIFS -- Vice de motifs (défaut / contradiction)
Définition : tout jugement doit être motivé (article 455 du code de procédure
civile, l'article signature de cette famille). Le défaut porte ici sur la
motivation elle-même, pas sur le fond du droit : l'arrêt ne répond pas à un
argument, se contredit, doute, ou ne motive pas.

6. DENATURATION -- Dénaturation
Définition : un contrat, un testament, des conclusions peuvent avoir un sens
clair et précis. Le juge du fond est souverain pour interpréter ce qui est
ambigu. S'il donne à un écrit clair un sens contraire à ses termes, la Cour
casse. Fondement : article 1192 du code civil (ancien article 1134), obligation
pour le juge de ne pas dénaturer l'écrit qui lui est soumis.

7. MBL -- Manque de base légale
Définition : le manque de base légale ne dit pas que la décision est fausse. Il
dit qu'elle est invérifiable : les faits constatés ne suffisent pas à contrôler
que la règle de droit a été bien appliquée. Le cas typique est celui du juge qui
a omis de rechercher un fait déterminant.

8. VIOLATION -- Violation de la loi
Définition : l'arrêt a mal appliqué la loi. Il a refusé d'appliquer une règle,
l'a mal interprétée, ou l'a appliquée à une situation qu'elle ne vise pas. C'est
la famille la plus générique. Doctrine : un seul cas d'ouverture, trois formes
(refus d'application, fausse interprétation, fausse application).

9. autre_indetermine -- Autre / Indéterminé
Définition : aucune des huit familles ci-dessus ne correspond avec certitude au
grief retenu par la Cour (par exemple une cassation par voie de conséquence sans
grief propre dans l'extrait, un extrait qui n'expose pas lui-même de grief de
cassation, ou un cas de cassation réel qui n'entre dans aucune des huit
familles).

Règle de départage : une seule famille par extrait. Si plusieurs familles
semblent applicables, retiens la plus spécifique, c'est-à-dire la première dans
l'ordre 1 à 8 ci-dessus dont le signal est réellement présent dans l'extrait.
VIOLATION (8) est la famille générique par défaut : ne la retiens que si aucune
famille plus spécifique (1 à 7) ne s'applique. N'utilise autre_indetermine (9)
que si aucune des huit familles doctrinales ne correspond.

Fonde ta réponse sur le raisonnement juridique décrit dans l'extrait, pas sur une
formule de surface attendue.

Voici l'extrait de motivation à classer :
---
{{MOTIVATION}}
---

Réponds uniquement par un objet JSON strictement de la forme suivante, sans
aucun texte avant ou après :
{"famille": "<UN_CODE_PARMI_LES_NEUF>"}

où <UN_CODE_PARMI_LES_NEUF> est exactement l'un des neuf codes suivants :
OMISSION_ULTRA_PETITA, EXCES_OFFICE, VICE_FORME, CONTRARIETE_JUGEMENTS,
VICE_MOTIFS, DENATURATION, MBL, VIOLATION, autre_indetermine.
```
<!-- PROMPT_ACCEPTE_V1_END -->

## Prompt, rejected side (fondement du rejet)

<!-- PROMPT_REJETE_V1_START -->
```text
Tu es un juriste spécialisé en procédure civile française. Tu vas lire un extrait de motivation d'un arrêt de la Cour
de cassation qui REJETTE un moyen, et déterminer sur quel(s) fondement(s) la
Cour le rejette.

Contexte doctrinal : quand la Cour de cassation rejette un moyen, elle le fait de
l'une de trois manières. Ces trois manières sont les trois familles à annoter
(référence doctrinale : Boré et Boré, La cassation en matière civile, Dalloz
Action, 6e éd. 2023-2024).

Trois familles possibles, à évaluer chacune indépendamment (plusieurs peuvent
s'appliquer à la fois), plus une catégorie résiduelle :

1. RNSM -- Rejet non spécialement motivé
Définition : depuis 2014, la Cour peut écarter un moyen qui n'est manifestement
pas de nature à entraîner la cassation, sans motivation spéciale. La réponse est
une phrase standard, sans explication. Article signature : article 1014 du code
de procédure civile (alinéa 1 : tout le pourvoi, alinéa 2 : certains moyens
seulement, le reste étant motivé).

2. IRREC -- Irrecevabilité / inopérance
Définition : avant d'examiner si un moyen a raison, la Cour vérifie qu'il peut
être présenté. Est écarté sans examen au fond le moyen qui invoque un argument
jamais soumis aux juges du fond, qui contredit ce que la partie soutenait devant
eux, qui critique un motif sans incidence sur la décision, ou qui demande à la
Cour de rejuger les faits.

3. FOND -- Rejet au fond
Définition : la Cour examine le moyen et le rejette en expliquant pourquoi la
cour d'appel a bien jugé.

4. Indéterminé -- aucune des trois familles ci-dessus ne s'applique avec
certitude (par exemple un rejet sans aucune formule reconnaissable, une
paraphrase pure du raisonnement, ou un rejet implicite).

Règle de réponse : ce n'est PAS un choix unique. Évalue les trois familles RNSM,
IRREC et FOND indépendamment l'une de l'autre, dans cet ordre :
1) La réponse est-elle une phrase rituelle sans motivation spéciale (ou son visa
   article 1014) ? Si oui, retiens RNSM.
2) La Cour oppose-t-elle un obstacle de recevabilité, sans juger le fond ? Si
   oui, retiens IRREC.
3) Sinon, la Cour explique-t-elle pourquoi le moyen a tort ? Si oui, retiens
   FOND.
Un même extrait peut cumuler deux familles (par exemple une phrase rituelle sur
une branche du moyen et un rejet motivé sur une autre branche) : retiens alors
toutes les familles réellement présentes. Ne force pas une famille unique. Si,
et seulement si, aucune des trois familles ne s'applique, indique
indetermine=true et laisse la liste familles vide.

Fonde ta réponse sur le raisonnement juridique décrit dans l'extrait, pas sur une
formule de surface attendue.

Voici l'extrait de motivation à classer :
---
{{MOTIVATION}}
---

Réponds uniquement par un objet JSON strictement de la forme suivante, sans
aucun texte avant ou après :
{"familles": [<LISTE PARMI "RNSM", "IRREC", "FOND">], "indetermine": <true ou false>}

où la liste "familles" contient zéro, une, deux ou trois valeurs parmi RNSM,
IRREC, FOND (jamais autre chose), et "indetermine" vaut true uniquement si la
liste "familles" est vide.
```
<!-- PROMPT_REJETE_V1_END -->

# The two rule sets

The classification rests on two rule sets, one per side of a pair. Both are
plain regex collections, read by the pipeline and never modified at run time.

## detectors.py, for accepted grounds

When the Court quashes, this file decides which doctrinal family the cassation
belongs to (violation of the law, lack of legal basis, distortion, and so on).
Its regexes look for the markers of each family in the Court's block of
reasons and are applied in a fixed order, so that a specific family always
wins over a generic one. The families follow Boré, *La cassation en matière
civile* (chapters 71 to 79).

## grille.json, for rejected grounds

When the Court rejects, this file identifies the formula that closed the
ground and maps it to a rejection code. The codes are numbered R1 to R12:
R1 is the summary rejection without specific reasons, the codes starting
with R2 and R4 mark inadmissibility, and the rest reject on the merits.
A rejected ground can carry several codes at once.

## How the two combine

A ground matched only by the first rule set is accepted. Matched only by the
second, it is rejected. Matched by both, an arbitration rule decides. Matched
by neither, it is labelled unmatched. The rarer outcomes (cassation by mere
consequence of another ground, reasons that are not a response on the merits,
conflicts) come out of the same arbitration.

The pattern names inside both files are published data, written
verbatim into the dataset's label columns. Renaming any of them is a breaking
change for every downstream reuse.

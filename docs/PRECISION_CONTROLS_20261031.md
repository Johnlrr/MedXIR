# Precision-first follow-up to 0.2916

User priority: bring Doc and Chunk precision towards 0.30 while preserving as much
recall as possible. Baseline macro P/R: Doc 0.1593/0.5650; Chunk 0.1901/0.3610.
No achieved precision or predicted leaderboard score is claimed for these controls.

## Immediate controls

Source: `../submission/medxir-v2ens-bgefull-drr-d200-c120-.zip`, SHA256
`0136db763db3442f5df02238a948618ff71f8b64a62e142a80ceebd75f86c4be`.

`scripts/build_precision_controls.py` streams one query at a time, preserves source
text and types, writes separate BZIP2 ZIPs, verifies full readback and source hash,
and refuses to overwrite outputs. Outputs and per-query counts are in
`../submission/precision-controls-20261031/precision_controls_manifest.json`.

- `d150-protected`: first 150 submitted docs plus every doc referenced by a retained
  chunk. All chunks unchanged. Actual doc count may exceed 150. This isolates doc
  selection; Chunk metrics should stay unchanged on the same evaluation set.
- `c80-prefix`: keep texts represented by the first 80 submitted chunk entries and
  all their fanout copies. All docs unchanged. Actual chunk count may exceed 80.
  This isolates chunk selection; Doc metrics should stay unchanged.

These are selection-order controls, NOT score-calibrated filters. The ZIP does not
contain scores, and the pipeline can output coverage-first windows followed by
score-ordered windows. A prefix must not be represented as top-80 by BGE confidence.
Neither variant is automatically recommended for a scarce submission slot.

## Required next evidence

The user has been asked for the exact 0.2916 output/pool/full-BGE-score link.
Available local `best_pool` belongs to 0.2755 and cannot silently stand in for the
new full-BGE scores. Origin/main remains ed1a4ac at the latest check.

With exact scores: bind query, chunk/doc IDs and source text; audit score coverage
and actual selection order; compare removing the low-scoring tail versus the
prefix controls. Do not equate sigmoid/min-max scores with calibrated probability.
Keep model weights fixed initially. No arbitrary language exclusion or query-ID
specific rules. Keep short but legitimate answer-bearing passages.

Qwen full job is RUNNING at the latest check. It scores different windows from the
baseline. Match source offsets/text before using its scores; do not join only on
doc ID and pretend scores apply to every window of that doc. Model disagreement
is not evidence that the Qwen-only candidate is correct.

Before proposing a submission: check 1200 query IDs, unchanged source text, IDs,
doc/chunk integrity, safe fanout, exact duplicates, ZIP bytes, and manifest hashes.
Use a predeclared sample across languages/query lengths for qualitative inspection;
such review is not gold evaluation and cannot estimate macro precision reliably.

Select the next experiment after those checks. Require user approval before using
a leaderboard attempt. Record all official Doc/Chunk P/R/F2, not only FINAL. A
precision gain accompanied by lower F2 is a tradeoff, not an overall improvement.
Do not open new embedding jobs or re-chunk the entire corpus for these experiments.

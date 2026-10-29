# Risk Scoring and Prioritization — Design & Decisions (Issue #16)

**Status:** implemented on branch `16-Risk-Scores-and-Prioritization`. This file
records the agreed approach and the decisions behind it (the issue's "record the
decisions" item); the code that realises it is listed under
[What shipped](#what-shipped), and [Handoff](#handoff-to-the-next-issue-dbstreamlit)
maps the remaining DB↔Streamlit work to the next issue.

## Deliverable

For each eligible member, produce:

1. a **risk score** from the trained model,
2. a **short, readable reason** for that score (from SHAP), and
3. a **priority** that orders the outreach worklist, with critical indicators
   (self-harm / suicidal thoughts) forced to the top.

These are written to the database so the Prioritization and Recommendation
stages can read them.

## Inputs reused (nothing new needed upstream)

| Need | Already provided by |
| --- | --- |
| A fitted model that bundles preparation | `artifacts/<name>/model.joblib` from [`models/train.py`](../src/mh_risk_outreach/models/train.py) |
| Identical train/score preparation | [`build_preparation_pipeline`](../src/mh_risk_outreach/prepare.py), bundled inside the saved pipeline |
| The 8000/2000 split, reproducible (seed 42) | [`datasets.split_dataset`](../src/mh_risk_outreach/datasets.py) |
| The three models | `REGISTRY` in [`models/__init__.py`](../src/mh_risk_outreach/models/__init__.py) |
| Per-model table pattern to copy | [`db/results.py`](../src/mh_risk_outreach/db/results.py) (`_safe_table`, schema-first + one transaction) |
| Critical-indicator override rule | Already specified in [ml-spec.md](ml-spec.md) Scoring step 4 |

## Decisions

### D1 — Score all three models; score the held-out 2000

Each model is trained on the 8000 training rows (the existing stratified split)
and then scores the **2000 evaluation rows it never saw** — an honest
demonstration rather than scoring rows the model memorised. `split_dataset`
already yields exactly those 8000 / 2000 rows for seed 42, so no new split
logic is added.

The scored set is a parameter (default = the evaluation split) so the same code
can later score *all* eligible members or a fresh batch without change.

> Note: training writes one results row per run today; scoring is a separate,
> additive step that reuses the fitted pipeline. We can either score straight
> from `train_and_evaluate`'s in-memory pipeline or reload the saved
> `model.joblib`; reloading is preferred so a stored score always traces to a
> saved `model_version`.

### D2 — Reasons from SHAP, one explainer per model family

`shap` is added as a dependency. One engine covers all three models:

| Model | Explainer | For the linear model, equals |
| --- | --- | --- |
| `logistic_regression` | `shap.LinearExplainer` | signed weight × (value − training mean) — the exact "top reasons" story already in ml-spec.md |
| `random_forest` | `shap.TreeExplainer` | — |
| `xgboost` | `shap.TreeExplainer` | — |

SHAP returns a signed contribution per **prepared** column (44 of them) for each
member. To turn those into a reason we:

1. take the contributions that pushed the score **up** (most positive),
2. map each prepared column back to its source feature and a human phrase
   (e.g. `Work_Stress_Level` high → "high work stress"; `Social_Support` low →
   "low social support"; a one-hot column → "employment: unemployed"), and
3. join the top 2–3 into one short sentence stored in `reasoning`.

A small phrase table (feature → how to describe a high / low / set value) lives
with the scoring code. This is the bulk of the new work and the part the issue
calls "turn those contributions into reasons the team can understand."

### D3 — Risk levels via K-means on the scores (5 bands)

Five levels — **Very High, High, Medium, Low, Very Low** — are assigned by
running 1-D K-means (`k=5`, fixed `random_state`) on that model's scores, then
ordering the five clusters by centroid and labelling them Very Low → Very High.

Why K-means rather than fixed cutoffs or fixed percentages:

- It **learns the band boundaries from the data**, which is exactly the issue's
  "without choosing thresholds or priority bands in advance."
- Fixed score cutoffs (e.g. 80–100 = HIGH) are explicitly warned against in
  ml-spec.md: with 92% of rows positive, ~97% land in HIGH and the worklist
  becomes useless.
- It is consistent with ml-spec.md's note that K-means belongs in **member
  segmentation**, not as the classifier — banding scores is exactly that.

Honest caveats (all manageable, all to be documented next to the numbers):

- Scores bunch near the top, so clusters can be uneven / small. Acceptable — it
  reflects the real distribution — but the band sizes must be reported.
- K-means is random-initialised: `random_state` is fixed for reproducibility.
- Bands are computed **per model**, so "High" in one table is not numerically
  the same as "High" in another. Documented, not hidden.
- Quantile bands (e.g. top 10 / 20 / 40 / 20 / 10) are the simple fallback if
  the clusters prove too lopsided to be useful.

### D4 — Priority ordering and the critical-indicator override

Priority is carried by **two columns**: `band_level_priority` (the band label)
and `rank_priority` (a strict numeric order, 1 = contact first).

1. **Critical indicators first.** Any member with `Self_Harm_Thoughts = 1` or
   `Suicidal_Thoughts = 1` is forced to **Very High** in `band_level_priority`
   and sorted above everyone else in `rank_priority`, regardless of model score
   — the safety override already in ml-spec.md. `risk_score` is left at its raw
   model value so the override is visible (a low score but top priority), and
   the `reasoning` text names the critical indicator.
2. Everyone else takes the band their score falls in (D3), ordered by band then
   by raw score within the band.
3. `rank_priority` ties are broken by `risk_score` then `Member_ID` so the order
   is fully reproducible.

### D5 — Output: three tables, one per model

Mirroring the per-model results tables, scoring writes `logistic_regression_scores`,
`random_forest_scores`, and `xgboost_scores` through one shared writer modelled on
`db/results.py`. Re-running a model replaces that table's contents in one
transaction (like `repository.save_records`), so each table holds the current
worklist for its model. Columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `Member_ID` | INTEGER | the member (joins back to `patients`) |
| `model_version` | TEXT | which trained model produced the row (ml-spec versioning rule) |
| `risk_score` | REAL | model probability × 100, 0–100 (raw, never overridden) |
| `band_level_priority` | TEXT | the priority band: Very High … Very Low (K-means, override → Very High) |
| `rank_priority` | INTEGER | numeric contact order, 1 = contact first (override pins criticals to the top) |
| `reasoning` | TEXT | short SHAP-based description of the top drivers |

### D6 — `risk_score` vs `risk_level` vs `priority` (so the three stay distinct)

| Field | What it is | Example |
| --- | --- | --- |
| `risk_score` | continuous model output, 0–100 | `87.4` |
| risk level | the K-means band the score falls in | `High` |
| `priority` | the contact order / tier after the override | `Very High` (critical) or rank `1` |

A member can have a high *score* but be pushed to the top *priority* only by the
override; score and priority are not the same column.

## Planned module layout

```
src/mh_risk_outreach/
  scoring.py          # load model -> risk_score, risk level (K-means), priority
  reasons.py          # SHAP contributions -> readable `reasoning` text
  db/scores.py        # per-model scores table writer (copies db/results.py)
tests/unit/
  test_scoring.py     # see testing plan
docs/
  ml-spec.md          # Scoring / Interpreting / Priority sections updated to SHAP + K-means
```

A CLI entry (`python -m mh_risk_outreach.scoring --model <name>`) scores a model
and writes its table, matching how `models.train` is run today. A Streamlit page
can show the resulting worklist later; not required for this issue.

## Testing plan (mirrors `tests/unit/test_models.py` style)

- Every `risk_score` is in 0–100, exactly one row per scored member.
- A member with a critical indicator always lands at top priority.
- A row with every symptom at maximum scores higher than the same row at zero.
- `reasoning` is non-empty and names real source features.
- A saved → reloaded model produces identical scores (reuses the existing
  determinism guarantee).
- A fixed ~20-row sample keeps the same ranking after a code change (ordering
  stability).
- K-means banding is reproducible for a fixed seed and yields ≤ 5 ordered bands.

## Settled decisions (team review, this branch)

1. **Priority is two columns** — `band_level_priority` (band label) **and**
   `rank_priority` (numeric rank). ✔
2. **`model_version` column is kept** on every table, per the versioning rule. ✔
3. **Table naming:** `<model>_scores`. ✔
4. **K-means `k = 5`** for the five named levels, with quantile bands as the
   documented fallback if clusters prove too lopsided. ✔
5. **Docs/READMEs updated** alongside the implementation (ml-spec.md Scoring /
   Interpreting / Priority, docs index, and the top-level README). ✔

## What shipped

| Piece | File | Role |
| --- | --- | --- |
| Scoring + bands + override + ranks | [`scoring.py`](../src/mh_risk_outreach/scoring.py) | `score_members` builds the worklist; `assign_bands` does the K-means; a `--model <name>\|all --scope evaluation\|all` CLI |
| SHAP → readable reasons | [`reasons.py`](../src/mh_risk_outreach/reasons.py) | `build_reasons`; `LinearExplainer` for LR, `TreeExplainer` for RF/XGBoost; per-feature phrase table |
| Per-model scores table | [`db/scores.py`](../src/mh_risk_outreach/db/scores.py) | `save_member_scores` / `load_member_scores` / `scores_status`; one `<model>_scores` table, replaced per run |
| Tests | [`tests/unit/test_scoring.py`](../tests/unit/test_scoring.py) | score range, dense rank, override, monotonicity, readable reasons, reproducible bands, determinism, round-trip, status |

Verified end to end: all three tables hold 2,000 scored evaluation members, the
342 critical members occupy the top ranks as Very High, and `rank_priority` is a
dense 1..N in each table.

## Handoff to the next issue (DB ↔ Streamlit)

The next issue connects this work to the Milestone 1 database and Streamlit app.
Its storage half is already done here; what it adds is the interface half.

| Next-issue checklist item | State after this branch |
| --- | --- |
| Store scores, reasons, levels, priority, model version | **Done** — `db/scores.py` writes all six columns per `<model>_scores` table |
| Save results against the correct member IDs | **Done** — rows are keyed by `Member_ID` and join back to `patients` |
| A clear way to run scoring / refresh results | **Partly** — the `scoring` CLI exists; a Streamlit "Run / refresh" control is the next issue |
| Display the prioritized table with readable scores, levels, reasons | **To do** — a new Streamlit page, alongside `ui/pages/1_Model_evaluation.py` |
| Handle missing results and failed runs with clear messages | **Primitive ready** — `scores_status(model)` returns `(exists, row_count)` so the page can show "run scoring first" without crashing, mirroring `database_status` |
| Check displayed results match the DB; document the workflow | **To do** — next issue, reading through `load_member_scores` |

Recommended entry point for the next branch: a `ui/pages/2_Member_worklist.py`
that uses `scores_status` + `load_member_scores`, with a button that calls
`scoring.run(model, scope=...)` to refresh.

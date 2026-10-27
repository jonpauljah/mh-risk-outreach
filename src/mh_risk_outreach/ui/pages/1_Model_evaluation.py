"""Streamlit page: train the registered risk models and compare them visually.

Trains every model in ``REGISTRY`` on the issue #14 split through the same
``train_and_evaluate`` the CLI uses, then charts the held-out scores: the checks
the unit tests assert, headline metrics next to the majority baseline, ranking
curves, positive rate by score decile, calibration, score distributions, and the
logistic-regression weights. Nothing is saved; use ``models.train`` for that.

Streamlit lists this page automatically beside ``ui/app.py``.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from mh_risk_outreach.config import DEFAULT_DB_PATH, DEFAULT_TABLE, TARGET
from mh_risk_outreach.datasets import assert_disjoint, split_dataset
from mh_risk_outreach.db.repository import database_status, load_from_db
from mh_risk_outreach.db.results import load_model_results
from mh_risk_outreach.evaluation import (
    DECISION_THRESHOLD,
    evaluate_scores,
    majority_baseline,
)
from mh_risk_outreach.models import DEFAULT_MODEL, REGISTRY
from mh_risk_outreach.models.train import train_and_evaluate

IMPORT_COMMAND = "uv run python -m mh_risk_outreach.etl.pipeline"

# Categorical slots 1-3 of the validated chart palette, fixed per model so a
# model keeps its color when others are hidden. Baseline/reference marks are gray.
MODEL_COLORS = {
    "logistic_regression": "#2a78d6",
    "random_forest": "#eb6834",
    "xgboost": "#1baf7a",
}
FALLBACK_COLOR = "#6250d6"
BASELINE_COLOR = "#8a8984"
# Diverging pair for signed weights: red raises the risk score, blue lowers it.
RAISES_COLOR, LOWERS_COLOR = "#e34948", "#2a78d6"

METRIC_LABELS = {
    "pr_auc": "PR-AUC (gate)",
    "roc_auc": "ROC-AUC",
    "brier": "Brier (lower is better)",
    "recall": "Recall",
    "precision": "Precision",
    "negative_recall": "Neg. recall",
    "accuracy": "Accuracy",
}


@st.cache_data
def _load(db_path: str, table: str) -> pd.DataFrame:
    return load_from_db(db_path, table)


@st.cache_resource(show_spinner="Training models on the 80% training split...")
def _train(model_name: str, _records: pd.DataFrame, row_count: int):
    """Fit one model and score the held-out rows; cached per model and data size."""
    result = train_and_evaluate(REGISTRY[model_name](), _records)
    evaluation = split_dataset(_records).evaluation
    y_prob = result.pipeline.predict_proba(evaluation)[:, 1]
    return result, evaluation[TARGET].to_numpy(dtype=int), y_prob


def _color_scale(names: list[str]) -> alt.Scale:
    return alt.Scale(
        domain=names,
        range=[MODEL_COLORS.get(name, FALLBACK_COLOR) for name in names],
    )


def _checks_table(scores: dict, split_ok: bool) -> pd.DataFrame:
    """The pass/fail checks from tests/unit/test_models.py, run on the full data."""
    rows = []
    for name, (_, y_true, y_prob) in scores.items():
        baseline = majority_baseline(y_true)
        metrics = evaluate_scores(y_true, y_prob)
        checks = {
            "Train/eval split disjoint": split_ok,
            "Scores are probabilities (0-1)": bool(np.all((y_prob >= 0) & (y_prob <= 1))),
            "PR-AUC beats baseline": metrics["pr_auc"] > baseline["pr_auc"],
            "Ranks better than chance (ROC-AUC > 0.5)": metrics["roc_auc"] > 0.5,
        }
        rows.append(
            {"Model": name, **{k: "✅ pass" if v else "❌ fail" for k, v in checks.items()}}
        )
    return pd.DataFrame(rows)


def _metrics_table(scores: dict, threshold: float) -> pd.DataFrame:
    rows = []
    for name, (_, y_true, y_prob) in scores.items():
        metrics = evaluate_scores(y_true, y_prob, threshold=threshold)
        rows.append({"Model": name, **{METRIC_LABELS[k]: metrics[k] for k in METRIC_LABELS}})
    baseline = majority_baseline(y_true)
    rows.append(
        {
            "Model": "majority baseline",
            METRIC_LABELS["pr_auc"]: baseline["pr_auc"],
            METRIC_LABELS["roc_auc"]: 0.5,
            METRIC_LABELS["brier"]: np.nan,
            METRIC_LABELS["recall"]: 1.0,
            METRIC_LABELS["precision"]: baseline["positive_rate"],
            METRIC_LABELS["negative_recall"]: 0.0,
            METRIC_LABELS["accuracy"]: baseline["accuracy"],
        }
    )
    return pd.DataFrame(rows)


def _curve_chart(curves: pd.DataFrame, reference: pd.DataFrame, x: str, y: str, names) -> alt.LayerChart:
    lines = (
        alt.Chart(curves)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(f"{x}:Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", scale=_color_scale(names), title="Model"),
            tooltip=["model:N", alt.Tooltip(f"{x}:Q", format=".3f"), alt.Tooltip(f"{y}:Q", format=".3f")],
        )
    )
    ref = (
        alt.Chart(reference)
        .mark_line(strokeWidth=1.5, strokeDash=[5, 4], color=BASELINE_COLOR)
        .encode(x=f"{x}:Q", y=f"{y}:Q", tooltip=[alt.Tooltip("label:N", title="Reference")])
    )
    return (ref + lines).properties(height=320)


def _decile_frame(scores: dict) -> pd.DataFrame:
    """Share of members with an issue in each tenth of the ranked scores."""
    frames = []
    for name, (_, y_true, y_prob) in scores.items():
        decile = pd.qcut(pd.Series(y_prob).rank(method="first"), 10, labels=range(1, 11))
        rate = pd.Series(y_true).groupby(decile, observed=True).mean()
        frames.append(pd.DataFrame({"model": name, "decile": rate.index.astype(int), "positive_rate": rate.values}))
    return pd.concat(frames)


def _calibration_frame(scores: dict) -> pd.DataFrame:
    """Mean predicted vs observed rate in ten equal-size score bins."""
    frames = []
    for name, (_, y_true, y_prob) in scores.items():
        bins = pd.qcut(pd.Series(y_prob).rank(method="first"), 10, labels=False)
        frame = pd.DataFrame({"pred": y_prob, "obs": y_true, "bin": bins})
        grouped = frame.groupby("bin").agg(predicted=("pred", "mean"), observed=("obs", "mean"))
        frames.append(grouped.assign(model=name))
    return pd.concat(frames)


def main() -> None:
    st.set_page_config(page_title="Model evaluation", layout="wide")
    st.title("Model evaluation")

    exists, row_count = database_status(DEFAULT_DB_PATH, DEFAULT_TABLE)
    if not exists or row_count == 0:
        st.warning(
            f"No records found in `{DEFAULT_DB_PATH}` (table `{DEFAULT_TABLE}`). "
            "Run the import to populate the database, then reload this page."
        )
        st.code(IMPORT_COMMAND, language="bash")
        st.stop()

    records = _load(DEFAULT_DB_PATH, DEFAULT_TABLE)
    names_all = list(REGISTRY)

    controls = st.columns([3, 2])
    names = controls[0].multiselect("Models", names_all, default=names_all)
    threshold = controls[1].slider(
        "Decision threshold (recall / precision / accuracy only)",
        0.05, 0.95, DECISION_THRESHOLD, 0.01,
    )
    if not names:
        st.info("Pick at least one model.")
        st.stop()

    scores = {name: _train(name, records, row_count) for name in names}
    split = split_dataset(records)
    try:
        assert_disjoint(split)
        split_ok = True
    except ValueError:
        split_ok = False
    y_true = next(iter(scores.values()))[1]
    baseline = majority_baseline(y_true)
    st.caption(
        f"Trained on {len(split.train):,} rows, evaluated on {len(split.evaluation):,} "
        f"held-out rows · {baseline['positive_rate']:.1%} of evaluation rows have an issue · "
        "nothing is saved from this page"
    )

    st.subheader("Checks")
    st.caption("The same assertions as `tests/unit/test_models.py`, run on the full dataset.")
    st.dataframe(_checks_table(scores, split_ok), hide_index=True, width="stretch")

    st.subheader("Headline metrics")
    st.caption(
        "PR-AUC is the gate: it must beat the majority baseline, which equals the share "
        f"of members with an issue. Threshold metrics use {threshold:.2f}."
    )
    table = _metrics_table(scores, threshold)
    st.dataframe(
        table.style.format(precision=3, na_rep="—"), hide_index=True, width="stretch"
    )

    st.subheader("How well each model ranks members")
    left, right = st.columns(2)
    pr_frames, roc_frames = [], []
    for name, (_, yt, yp) in scores.items():
        precision, recall, _ = precision_recall_curve(yt, yp)
        pr_frames.append(pd.DataFrame({"model": name, "recall": recall, "precision": precision}))
        fpr, tpr, _ = roc_curve(yt, yp)
        roc_frames.append(pd.DataFrame({"model": name, "false_positive_rate": fpr, "true_positive_rate": tpr}))
    with left:
        st.markdown("**Precision vs recall** — dashed line is the baseline (always \"yes\")")
        reference = pd.DataFrame(
            {"recall": [0, 1], "precision": [baseline["pr_auc"]] * 2, "label": "majority baseline"}
        )
        st.altair_chart(
            _curve_chart(pd.concat(pr_frames), reference, "recall", "precision", names),
            width="stretch",
        )
    with right:
        st.markdown("**ROC** — dashed diagonal is random ranking")
        reference = pd.DataFrame(
            {"false_positive_rate": [0, 1], "true_positive_rate": [0, 1], "label": "random"}
        )
        st.altair_chart(
            _curve_chart(pd.concat(roc_frames), reference, "false_positive_rate", "true_positive_rate", names),
            width="stretch",
        )

    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Share with an issue, by score decile** — 1 = lowest-scored tenth. "
            "A flat line means the ranking carries no signal."
        )
        deciles = _decile_frame(scores)
        lines = (
            alt.Chart(deciles)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64))
            .encode(
                x=alt.X("decile:O", title="Score decile"),
                y=alt.Y("positive_rate:Q", title="Share with an issue", scale=alt.Scale(zero=False), axis=alt.Axis(format="%")),
                color=alt.Color("model:N", scale=_color_scale(names), title="Model"),
                tooltip=["model:N", "decile:O", alt.Tooltip("positive_rate:Q", format=".1%")],
            )
        )
        rule = alt.Chart(pd.DataFrame({"y": [baseline["positive_rate"]]})).mark_rule(
            strokeDash=[5, 4], color=BASELINE_COLOR
        ).encode(y="y:Q")
        st.altair_chart((rule + lines).properties(height=320), width="stretch")
    with right:
        st.markdown(
            "**Calibration** — predicted vs observed rate in ten score bins. "
            "On the dashed diagonal = honest probabilities."
        )
        calibration = _calibration_frame(scores)
        diag = alt.Chart(pd.DataFrame({"predicted": [0.6, 1], "observed": [0.6, 1]})).mark_line(
            strokeDash=[5, 4], color=BASELINE_COLOR, strokeWidth=1.5
        ).encode(x="predicted:Q", y="observed:Q")
        points = (
            alt.Chart(calibration)
            .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=64))
            .encode(
                x=alt.X("predicted:Q", title="Mean predicted probability", scale=alt.Scale(zero=False)),
                y=alt.Y("observed:Q", title="Observed share with an issue", scale=alt.Scale(zero=False)),
                color=alt.Color("model:N", scale=_color_scale(names), title="Model"),
                tooltip=["model:N", alt.Tooltip("predicted:Q", format=".3f"), alt.Tooltip("observed:Q", format=".3f")],
            )
        )
        st.altair_chart((diag + points).properties(height=320), width="stretch")

    st.subheader("Score distribution by true label")
    focus = st.selectbox(
        "Model", names, index=names.index(DEFAULT_MODEL) if DEFAULT_MODEL in names else 0
    )
    _, yt, yp = scores[focus]
    st.caption(
        "Each panel is normalized to its own group. Heavily overlapping shapes = the "
        "model struggles to tell the two groups apart."
    )
    edges = np.linspace(yp.min(), yp.max(), 41)
    frames = []
    for label, mask in (("Has an issue", yt == 1), ("No issue", yt == 0)):
        counts, _ = np.histogram(yp[mask], bins=edges)
        frames.append(
            pd.DataFrame(
                {"label": label, "start": edges[:-1], "end": edges[1:], "share": counts / mask.sum()}
            )
        )
    hist = (
        alt.Chart(pd.concat(frames))
        .mark_bar(color=MODEL_COLORS.get(focus, FALLBACK_COLOR), cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("start:Q", title="Predicted probability of an issue", scale=alt.Scale(zero=False)),
            x2="end:Q",
            y=alt.Y("share:Q", title="Share of group", axis=alt.Axis(format="%")),
            y2=alt.datum(0),
            row=alt.Row("label:N", title=None),
            tooltip=[
                "label:N",
                alt.Tooltip("start:Q", title="Score from", format=".2f"),
                alt.Tooltip("end:Q", title="Score to", format=".2f"),
                alt.Tooltip("share:Q", format=".1%"),
            ],
        )
        .properties(height=150)
    )
    st.altair_chart(hist, width="stretch")

    if "logistic_regression" in scores:
        st.subheader("Logistic regression weights")
        st.caption(
            "The strongest weights in each direction. Red pushes the risk score up, blue "
            "pulls it down; these become each member's top reasons at scoring time. "
            "One-hot categories (e.g. `Smoking_*`) compare with each other, not with zero."
        )
        pipeline = scores["logistic_regression"][0].pipeline
        weights = pd.DataFrame(
            {
                "column": pipeline.named_steps["prepare"].get_feature_names_out(),
                "weight": pipeline.named_steps["model"].coef_[0],
            }
        )
        top = pd.concat(
            [weights.nlargest(10, "weight"), weights[weights["weight"] < 0].nsmallest(8, "weight")]
        )
        top["direction"] = np.where(top["weight"] > 0, "raises risk", "lowers risk")
        bars = (
            alt.Chart(top)
            .mark_bar(cornerRadius=2)
            .encode(
                x=alt.X("weight:Q", title="Weight"),
                y=alt.Y("column:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["raises risk", "lowers risk"], range=[RAISES_COLOR, LOWERS_COLOR]),
                    title="Direction",
                ),
                tooltip=["column:N", alt.Tooltip("weight:Q", format=".3f"), "direction:N"],
            )
            .properties(height=24 * len(top))
        )
        st.altair_chart(bars, width="stretch")

    st.subheader("Saved runs")
    history = []
    for name in names:
        try:
            history.append(load_model_results(name).assign(model=name))
        except pd.errors.DatabaseError:  # no results table yet for this model
            continue
    if history:
        st.caption("Rows written by `models.train` to each model's results table.")
        st.dataframe(pd.concat(history), hide_index=True, width="stretch")
    else:
        st.caption(
            "No saved runs yet. Record one with "
            "`uv run python -m mh_risk_outreach.models.train --model <name>`."
        )


main()

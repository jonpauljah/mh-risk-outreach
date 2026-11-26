"""Build manager worklists and enforce outreach eligibility when selected."""

import json
from collections.abc import Iterable
from contextlib import closing
from dataclasses import dataclass, field
from sqlite3 import Connection

from mh_risk_outreach.config import CRITICAL_INDICATORS, DEFAULT_DB_PATH, FEATURES
from mh_risk_outreach.db.assessments import _connect
from mh_risk_outreach.db.operational_population import load_operational_member_ids
from mh_risk_outreach.db.operational_scores import (
    WRONG_POPULATION,
    _published_cohort,
    worklist_requires_refresh,
)
from mh_risk_outreach.db.outreach import _create_case
from mh_risk_outreach.models.artifacts import (
    DEFAULT_ARTIFACTS_DIR,
    artifact_model_version,
)
from mh_risk_outreach.operational import (
    operational_model_name,
    score_operational_member,
)
from mh_risk_outreach.recommendations import SAFETY_REVIEW_ACTION, recommend_action


@dataclass(frozen=True)
class OutreachCandidate:
    """A member's current worklist entry and the provenance of a selection."""

    member_id: int
    assessment_id: int | None
    cohort_id: str | None
    band: str | None
    rank: int | None
    safety_flags: tuple[str, ...]
    pending_scoring: bool
    selectable: bool
    # The fields above identify the selection rechecked by create_cases().
    risk_score: float | None = field(default=None, compare=False)
    explanation: str | None = field(default=None, compare=False)
    recommendation_action: str | None = field(default=None, compare=False)
    recommendation_reason: str | None = field(default=None, compare=False)
    has_active_case: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class OutreachWorklists:
    active_cases: int
    standard_slots: int
    safety_review: tuple[OutreachCandidate, ...]
    standard_outreach: tuple[OutreachCandidate, ...]
    requires_score_refresh: bool = False
    scoring_issue: str | None = None


# Use live input flags instead of cached flags on the published score. A manual
# submission can report a new concern before any rescore has run.
_LATEST_FLAGS = ", ".join(
    f"COALESCE(json_extract(a.inputs_json, '$.{flag}'), p.{flag}) AS {flag}"
    for flag in CRITICAL_INDICATORS
)
_CANDIDATES = f"""
    SELECT p.Member_ID, state.assessment_id, a.source, a.scoring_status,
           a.recommendation_action, a.recommendation_reason,
           score.cohort_id, score.band_level_priority, score.rank_priority,
           score.risk_score, score.reasoning,
           {_LATEST_FLAGS},
           EXISTS (
               SELECT 1 FROM outreach_cases AS active WHERE active.Member_ID = p.Member_ID
               AND active.status IN ('Pending', 'Accepted')
           ) AS has_active_case,
           EXISTS (
               SELECT 1 FROM outreach_cases AS used WHERE used.assessment_id = state.assessment_id
           ) AS used_assessment,
           (
               SELECT MAX(assessment_id) FROM outreach_cases AS completed
               WHERE completed.Member_ID = p.Member_ID AND completed.status = 'Completed'
           ) AS last_completed_assessment
    FROM patients AS p
    JOIN json_each(?) AS operational ON operational.value = p.Member_ID
    LEFT JOIN member_assessment_state AS state ON state.Member_ID = p.Member_ID
    LEFT JOIN risk_assessments AS a ON a.assessment_id = state.assessment_id
    LEFT JOIN operational_scores AS score ON score.Member_ID = p.Member_ID
         AND score.operational_model_name = ? AND score.model_version = ?
         AND score.score_assessment_id IS state.assessment_id
         AND score.safety_assessment_id IS state.assessment_id
         AND score.inputs_pending_scoring = 0
         AND (state.assessment_id IS NULL OR (
             a.scoring_status = 'Scored'
             AND a.operational_model_name = score.operational_model_name
             AND a.model_version = score.model_version
         ))
    ORDER BY score.rank_priority NULLS LAST, p.Member_ID
"""


@dataclass(frozen=True)
class OutreachPolicy:
    """Use the same configured capacity for listing and creating outreach cases."""

    capacity: int
    db_path: str = DEFAULT_DB_PATH
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity < 0:
            raise ValueError("outreach capacity must be a nonnegative integer")

    def worklists(self) -> OutreachWorklists:
        """Read both queues and the shared capacity from one SQLite snapshot."""
        with closing(_connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN")
            return self._worklists(conn)

    def _worklists(self, conn: Connection) -> OutreachWorklists:
        active = conn.execute(
            "SELECT COUNT(*) FROM outreach_cases WHERE status IN ('Pending', 'Accepted')"
        ).fetchone()[0]
        slots = max(0, self.capacity - active)
        member_ids = load_operational_member_ids(conn)
        requires_refresh = worklist_requires_refresh(conn, member_ids)
        model, version, scoring_issue = self._scoring_context(conn, requires_refresh)
        safety: list[OutreachCandidate] = []
        standard: list[OutreachCandidate] = []
        for row in conn.execute(
            _CANDIDATES, (json.dumps(sorted(member_ids)), model, version)
        ):
            if row["used_assessment"]:
                continue
            flags = tuple(flag for flag in CRITICAL_INDICATORS if row[flag] == 1)
            if not flags and (
                len(standard) >= slots
                or row["band_level_priority"] not in ("High", "Very High")
            ):
                continue

            assessment_id = row["assessment_id"]
            completed_id = row["last_completed_assessment"]
            action = row["recommendation_action"]
            selectable = (
                not row["has_active_case"]
                and row["cohort_id"] is not None
                and (
                    assessment_id is None
                    or (
                        bool(action) and (action == SAFETY_REVIEW_ACTION) == bool(flags)
                    )
                )
            )
            if not flags and (
                not selectable
                or (
                    completed_id is not None
                    and (
                        row["source"] != "reassessment" or assessment_id <= completed_id
                    )
                )
            ):
                continue

            reason = row["recommendation_reason"]
            if flags and action != SAFETY_REVIEW_ACTION:
                recommendation = recommend_action(
                    {flag: row[flag] for flag in CRITICAL_INDICATORS}
                )
                action = recommendation["recommendation_action"]
                reason = recommendation["recommendation_reason"]

            candidate = OutreachCandidate(
                member_id=row["Member_ID"],
                assessment_id=assessment_id,
                cohort_id=row["cohort_id"],
                band=row["band_level_priority"],
                rank=row["rank_priority"],
                safety_flags=flags,
                pending_scoring=row["scoring_status"] == "PendingScoring",
                selectable=selectable,
                risk_score=row["risk_score"],
                explanation=row["reasoning"],
                recommendation_action=action,
                recommendation_reason=reason,
                has_active_case=bool(row["has_active_case"]),
            )
            (safety if flags else standard).append(candidate)
        return OutreachWorklists(
            active,
            slots,
            tuple(safety),
            tuple(standard),
            requires_refresh,
            scoring_issue,
        )

    def _scoring_context(
        self, conn: Connection, requires_refresh: bool
    ) -> tuple[str | None, str | None, str | None]:
        if requires_refresh:
            return None, None, WRONG_POPULATION
        try:
            model = operational_model_name()
            version = artifact_model_version(model, self.artifacts_dir)
        except (TypeError, ValueError) as exc:
            return None, None, str(exc)
        if version is None:
            return None, None, f"Train the configured operational model ({model})."
        cohort = _published_cohort(conn)
        if cohort is None or (cohort.model_name, cohort.model_version) != (
            model,
            version,
        ):
            return None, None, "Publish scores for the configured operational model."
        return model, version, None

    def create_case(self, selection: OutreachCandidate) -> int:
        """Create a single case using the same policy as a bulk selection."""
        return self.create_cases((selection,))[0]

    def create_cases(self, selections: Iterable[OutreachCandidate]) -> list[int]:
        """Atomically create Pending cases and return IDs in selection order."""
        selected = tuple(selections)
        if not selected:
            return []
        if len({candidate.member_id for candidate in selected}) != len(selected):
            raise ValueError("a member can be selected only once")

        first_cases = [
            candidate for candidate in selected if candidate.assessment_id is None
        ]
        baselines = {}
        if first_cases:
            # Calculate recommendations outside the write transaction, with
            # imported inputs from the same snapshot as the eligibility check.
            with closing(_connect(self.db_path)) as conn, conn:
                conn.execute("BEGIN")
                self._require_selections(conn, selected)
                inputs_by_member = {}
                for candidate in first_cases:
                    member = conn.execute(
                        "SELECT * FROM patients WHERE Member_ID=?",
                        (candidate.member_id,),
                    ).fetchone()
                    inputs_by_member[candidate.member_id] = {
                        name: member[name] for name in FEATURES
                    }
            for candidate in first_cases:
                baseline = score_operational_member(
                    candidate.member_id,
                    inputs_by_member[candidate.member_id],
                    db_path=self.db_path,
                    artifacts_dir=self.artifacts_dir,
                )
                context = baseline["rank_context"]
                if (
                    not isinstance(context, dict)
                    or context.get("cohort_id") != candidate.cohort_id
                ):
                    raise ValueError(
                        "operational scoring changed; refresh the outreach worklists"
                    )
                baselines[candidate.member_id] = baseline

        with closing(_connect(self.db_path)) as conn, conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_selections(conn, selected)
            return [
                self._persist_selection(
                    conn, candidate, baselines.get(candidate.member_id)
                )
                for candidate in selected
            ]

    @staticmethod
    def _persist_selection(
        conn: Connection,
        selection: OutreachCandidate,
        baseline: dict[str, object] | None,
    ) -> int:
        if baseline is not None:
            return _create_case(
                conn,
                selection.member_id,
                str(baseline["recommendation_action"]),
                recommendation_reason=str(baseline["recommendation_reason"]),
                recommendation_rule_version=str(
                    baseline["recommendation_rule_version"]
                ),
            )
        action = conn.execute(
            "SELECT recommendation_action FROM risk_assessments WHERE assessment_id=?",
            (selection.assessment_id,),
        ).fetchone()[0]
        return _create_case(conn, selection.member_id, action)

    def _require_selections(
        self, conn: Connection, selections: tuple[OutreachCandidate, ...]
    ) -> None:
        worklists = self._worklists(conn)
        if worklists.requires_score_refresh:
            raise ValueError(WRONG_POPULATION)
        standard_count = sum(not selection.safety_flags for selection in selections)
        if standard_count > worklists.standard_slots:
            raise ValueError("standard outreach is at capacity")
        selectable = set(worklists.standard_outreach)
        selectable.update(
            candidate for candidate in worklists.safety_review if candidate.selectable
        )
        if any(selection not in selectable for selection in selections):
            raise ValueError(
                "selection is no longer eligible; refresh the outreach worklists"
            )

"""Persist and retrieve versioned model bundles with frozen scoring bands."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple

import joblib
from sklearn.pipeline import Pipeline

from mh_risk_outreach.bands import BAND_LABELS, BAND_N_INIT, validate_cut_points
from mh_risk_outreach.config import RANDOM_STATE
from mh_risk_outreach.model_inputs import MODEL_FEATURES
from mh_risk_outreach.prepare import PREPARED_COLUMN_COUNT

DEFAULT_ARTIFACTS_DIR = "artifacts"


class ScoringArtifact(NamedTuple):
    model_name: str
    model_version: str
    pipeline: Pipeline
    cut_points: tuple[float, ...]


def _path_component(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or any(char.isspace() for char in value)
    ):
        raise ValueError("invalid artifact model name or version")
    return value


def _check_feature_contract(metadata: object, model_name: str) -> dict:
    if not isinstance(metadata, dict):
        raise TypeError(f"invalid metadata for {model_name}: expected an object")
    if (
        metadata.get("model_name") != model_name
        or metadata.get("model_features") != list(MODEL_FEATURES)
        or metadata.get("prepared_column_count") != PREPARED_COLUMN_COUNT
        or not metadata.get("model_version")
    ):
        raise ValueError(
            f"saved {model_name} artifact uses an outdated feature contract"
        )
    return metadata


def _check_bands(metadata: dict, version: str) -> tuple[float, ...]:
    if metadata["model_version"] != version:
        raise ValueError("artifact metadata does not match the requested model version")
    bands = metadata.get("risk_bands")
    if bands is None:
        raise ValueError("saved artifact is missing frozen risk bands")
    if not isinstance(bands, dict):
        raise TypeError("invalid frozen risk band metadata: expected an object")
    reference_count = bands.get("reference_row_count")
    if (
        bands.get("labels") != list(BAND_LABELS)
        or bands.get("reference_scope") != "train"
        or type(reference_count) is not int
        or reference_count < len(BAND_LABELS)
        or reference_count != metadata.get("n_train")
        or bands.get("algorithm") != "kmeans"
        or bands.get("random_state") != RANDOM_STATE
        or bands.get("n_init") != BAND_N_INIT
    ):
        raise ValueError("invalid frozen risk band metadata")
    return validate_cut_points(bands.get("cut_points"))


def _read_metadata(folder: Path, model_name: str) -> dict:
    try:
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid metadata for {model_name}: {exc}") from exc
    return _check_feature_contract(metadata, model_name)


def _resolve_bundle(
    model_name: str, artifacts_dir: str, version: str | None
) -> tuple[str, Path, tuple[float, ...]] | None:
    folder = Path(artifacts_dir) / _path_component(model_name)
    if version is None:
        try:
            version = (folder / "latest").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            if (folder / "model.joblib").exists() or (
                folder / "metadata.json"
            ).exists():
                _read_metadata(folder, model_name)
                raise ValueError(
                    f"saved {model_name} artifact predates frozen risk bands"
                )
            return None
        except OSError as exc:
            raise ValueError(
                f"cannot read the latest {model_name} version: {exc}"
            ) from exc
    version = _path_component(version)
    bundle = folder / "versions" / version
    cut_points = _check_bands(_read_metadata(bundle, model_name), version)
    model_path = bundle / "model.joblib"
    if not model_path.is_file():
        raise ValueError(f"missing model file for {model_name} version {version}")
    return version, model_path, cut_points


def artifact_model_version(
    model_name: str, artifacts_dir: str = DEFAULT_ARTIFACTS_DIR
) -> str | None:
    """Return the latest version when its metadata supports current scoring."""
    bundle = _resolve_bundle(model_name, artifacts_dir, None)
    return bundle[0] if bundle is not None else None


def load_scoring_artifact(
    model_name: str,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    *,
    version: str | None = None,
) -> ScoringArtifact:
    """Load the pipeline and cut points from the same versioned bundle."""
    train = f"`uv run python -m mh_risk_outreach.models.train --model {model_name}`"
    try:
        bundle = _resolve_bundle(model_name, artifacts_dir, version)
        if bundle is None:
            raise SystemExit(
                f"no saved model for {model_name}; train it first with {train}"
            )
        version, path, cut_points = bundle
        pipeline = joblib.load(path)
        if not isinstance(pipeline, Pipeline):
            raise TypeError(f"invalid pipeline for {model_name} version {version}")
    except Exception as exc:
        raise SystemExit(f"{exc}; retrain with {train}") from exc
    return ScoringArtifact(model_name, version, pipeline, cut_points)


def save_versioned_artifact(
    pipeline: Pipeline, metadata: dict[str, object], model_name: str, artifacts_dir: str
) -> Path:
    """Publish a complete immutable bundle before atomically advancing latest."""
    folder = Path(artifacts_dir) / _path_component(model_name)
    versions = folder / "versions"
    version = _path_component(
        _check_feature_contract(metadata, model_name)["model_version"]
    )
    _check_bands(metadata, version)
    versions.mkdir(parents=True, exist_ok=True)
    out = versions / version
    with TemporaryDirectory(dir=folder) as staging_dir:
        staging = Path(staging_dir)
        bundle = staging / "bundle"
        bundle.mkdir()
        joblib.dump(pipeline, bundle / "model.joblib")
        (bundle / "metadata.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
        )
        pointer = staging / "latest"
        pointer.write_text(f"{version}\n", encoding="utf-8")
        # A published destination is non-empty, so rename cannot overwrite it.
        bundle.rename(out)
        pointer.replace(folder / "latest")
    return out

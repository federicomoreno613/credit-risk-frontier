"""End-to-end parity tests between Kedro v1 and the curated thesis artifacts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[2]
KEDRO_OUT = ROOT / "data" / "08_reporting" / "kedro"


def _run_kedro_pipeline(pipeline_name: str) -> None:
    """Run a Kedro pipeline exactly as a user would from the repo root."""
    env = os.environ.copy()
    env.update(
        {
            "KEDRO_DISABLE_TELEMETRY": "1",
            "DO_NOT_TRACK": "1",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    result = subprocess.run(
        [sys.executable, "-m", "kedro", "run", "--pipeline", pipeline_name],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Falló kedro run --pipeline {pipeline_name}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


@pytest.fixture(scope="session")
def public_repro_outputs() -> Path:
    _run_kedro_pipeline("public_repro")
    return KEDRO_OUT


@pytest.fixture(scope="session")
def llm_local_outputs() -> Path:
    _run_kedro_pipeline("llm_local")
    return KEDRO_OUT


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _drop_kedro_wrapper(data: dict[str, Any]) -> dict[str, Any]:
    """Remove Kedro orchestration metadata before comparing with legacy artifacts."""
    cleaned = dict(data)
    cleaned.pop("source", None)
    return cleaned


def _assert_nested_equal(left: Any, right: Any, path: str = "root") -> None:
    """Compare nested JSON with float tolerance but strict keys/lists."""
    if isinstance(left, dict) and isinstance(right, dict):
        assert set(left) == set(right), f"Distintas claves en {path}: {set(left) ^ set(right)}"
        for key in left:
            _assert_nested_equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right), f"Distinto largo en {path}"
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            _assert_nested_equal(l_item, r_item, f"{path}[{idx}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12), f"Diferencia numérica en {path}"
        return
    assert left == right, f"Diferencia en {path}: {left!r} != {right!r}"


@pytest.mark.parametrize(
    ("legacy_path", "kedro_path"),
    [
        ("models/xgboost_metrics.json", "xgboost_metrics.json"),
        ("models/logreg_metrics.json", "logreg_metrics.json"),
        ("models/cv_seg_metrics.json", "cv_seg_metrics.json"),
    ],
)
def test_public_repro_metrics_match_legacy_json(public_repro_outputs: Path, legacy_path: str, kedro_path: str) -> None:
    legacy = _load_json(ROOT / legacy_path)
    kedro = _drop_kedro_wrapper(_load_json(public_repro_outputs / kedro_path))

    _assert_nested_equal(kedro, legacy)


def test_public_repro_comparison_table_matches_legacy_results(public_repro_outputs: Path) -> None:
    legacy = pd.read_csv(ROOT / "results" / "tabla_comparacion_final.csv")
    kedro = pd.read_csv(public_repro_outputs / "tabla_comparacion_final.csv")

    assert_frame_equal(kedro, legacy, check_dtype=False, rtol=1e-12, atol=1e-12)


def test_public_repro_dataset_contract_matches_known_thesis_counts(public_repro_outputs: Path) -> None:
    report = _load_json(public_repro_outputs / "dataset_validation_report.json")

    assert report["status"] == "ok"
    assert report["n_rows"] == 4897
    assert report["set_counts"] == {"train": 3910, "test": 495, "val": 492}
    assert report["segment_counts"] == {"denso": 1359, "esparso": 3538}
    assert report["test_segment_counts"] == {"denso": 291, "esparso": 204}
    assert report["expected_count_mismatches"] == []
    assert report["segment_rule"] == "(20 TU vars < 0).sum(axis=1) >= 6"


def test_public_repro_thesis_artifact_report_is_complete(public_repro_outputs: Path) -> None:
    report = _load_json(public_repro_outputs / "thesis_artifact_report.json")

    assert report["status"] == "ok"
    assert report["missing_files"] == []
    assert report["comparison_rows"] == len(pd.read_csv(ROOT / "results" / "tabla_comparacion_final.csv"))
    assert report["dataset_segments"] == {"denso": 1359, "esparso": 3538}
    for rel_path, check in report["required_file_checks"].items():
        assert check["exists"], rel_path
        assert check["bytes"] > 0, rel_path


def test_llm_local_reuses_existing_zero_shot_and_fewshot_artifacts(llm_local_outputs: Path) -> None:
    comparisons = [
        (ROOT / "models" / "llm_metrics_tabllm_nothink.json", llm_local_outputs / "llm_metrics_tabllm_nothink.json"),
        (ROOT / "models" / "llm_metrics_fewshot.json", llm_local_outputs / "llm_metrics_fewshot.json"),
    ]
    for legacy_path, kedro_path in comparisons:
        legacy = _load_json(legacy_path)
        kedro = _drop_kedro_wrapper(_load_json(kedro_path))
        _assert_nested_equal(kedro, legacy)


def test_llm_local_does_not_require_missing_thinking_artifact_or_ollama(llm_local_outputs: Path) -> None:
    """Thinking mode has no curated legacy artifact today, so the safe local run must skip it."""
    think_report = _load_json(llm_local_outputs / "llm_metrics_tabllm_think.json")

    assert not (ROOT / "models" / "llm_metrics_tabllm_think.json").exists()
    assert think_report["status"] == "skipped"
    assert think_report["source"] == "skipped"
    assert "activar llm_zero_shot.run_inference" in think_report["reason"]

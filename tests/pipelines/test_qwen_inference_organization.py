from pathlib import Path

import pandas as pd
import yaml

from credit_risk_frontier.pipelines.intermediate_delivery import qwen_inference


ROOT = Path(__file__).resolve().parents[2]


def _params() -> dict:
    configuration = yaml.safe_load(
        (ROOT / "conf/base/parameters_intermediate_delivery.yml").read_text()
    )
    return configuration["intermediate_qwen"]


def _features() -> list[str]:
    configuration = yaml.safe_load(
        (ROOT / "conf/base/parameters_intermediate_delivery.yml").read_text()
    )["intermediate_delivery"]["feature_contract"]
    return configuration["transunion"] + configuration["form_direct"]


def test_the_four_completed_fingerprints_are_frozen():
    dataset_sha = "ae4ed1e64456a4ff124c1ea789c66ef7d3b8f0e98e69c0d78bab9532ac08c90a"
    expected = {
        ("tu_form", 0): "10fd52e1d93759eca4a2db53896bfc272d47cb8b6bea983f540c99a936e32e1e",
        ("tu_form", 8): "42ba55dbd08acfc6a0e29ee3eee3064d0d0ec0dd224df00a3233711316808b17",
        ("tu_form_description", 0): "123d92d9c351f308f6714ca13bc8eaf51fc739525b58de0e6349bff56398acb9",
        ("tu_form_description", 8): "46ff6b12f961f1556487255cbbd1c51d81fa538089adff6bde9cdcc0e84d6cd1",
    }
    assert qwen_inference.experiment_configurations(_params()) == [
        ("tu_form", 0), ("tu_form", 8),
        ("tu_form_description", 0), ("tu_form_description", 8),
    ]
    for configuration, value in expected.items():
        assert qwen_inference.fingerprint(
            dataset_sha, *configuration, _features(), _params()
        ) == value

def test_status_counts_rows_without_running_inference():
    frame = pd.DataFrame({
        "credito_id_anon": ["a", "b", "c"],
        "set": ["test", "test", "train"],
    })
    params = _params()
    dataset_sha = "dataset"
    fp = qwen_inference.fingerprint(
        dataset_sha, "tu_form", 0, _features(), params
    )
    caches = {
        ("tu_form", 0): pd.DataFrame({
            "evaluation_id": ["a"],
            "valid": [True],
            "fingerprint": [fp],
        })
    }
    status = qwen_inference.summarize_cache_status(
        frame,
        caches,
        _features(),
        dataset_sha,
        params,
    )
    assert status["rows_present"] == 1
    assert status["rows_expected"] == 8
    tu_form_zero = next(
        item for item in status["configurations"]
        if item["profile"] == "tu_form" and item["shots"] == 0
    )
    assert tu_form_zero["first_pass_remaining"] == 1
    assert tu_form_zero["fingerprint_ok"] is True


def test_cli_uses_catalog_names_instead_of_physical_data_paths():
    script = (ROOT / "scripts/run_intermediate_qwen.py").read_text()
    assert 'catalog.load("model_input_table")' in script
    assert 'catalog.load("split_manifest")' in script
    assert "read_parquet" not in script
    assert "to_parquet" not in script

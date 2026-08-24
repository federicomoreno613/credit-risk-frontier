"""Ejecuta o consulta las cuatro inferencias recuperables de Qwen3-8B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from kedro.io import DatasetError

from credit_risk_frontier import utils
from credit_risk_frontier.pipelines.intermediate_delivery import qwen_inference


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Informa avance y huellas sin llamar al modelo.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Comprobación técnica no persistente. No modifica los cuatro caches "
            "canónicos ni produce resultados publicables."
        ),
    )
    return parser.parse_args()


def _load_optional(catalog, dataset_name: str) -> pd.DataFrame:
    try:
        return catalog.load(dataset_name)
    except (DatasetError, FileNotFoundError):
        return pd.DataFrame()


def main() -> None:
    args = parse_args()
    bootstrap_project(ROOT)
    with KedroSession.create(project_path=ROOT) as session:
        context = session.load_context()
        catalog = context.catalog
        frame = utils.annotate_segments(catalog.load("model_input_table"))
        split_manifest = catalog.load("split_manifest")
        delivery_params = context.params["intermediate_delivery"]
        qwen_params = context.params["intermediate_qwen"]
        features = {
            "tu_form": utils.intermediate_feature_columns(
                frame, "tu_form", delivery_params["feature_contract"]
            ),
            "tu_form_description": utils.intermediate_feature_columns(
                frame, "tu_form", delivery_params["feature_contract"]
            ),
        }
        dataset_sha = split_manifest["dataset_sha256"]

        caches = {
            configuration: _load_optional(catalog, dataset_name)
            for configuration, dataset_name
            in qwen_inference.CACHE_DATASETS.items()
        }
        if args.status:
            status = qwen_inference.summarize_cache_status(
                frame, caches, features, dataset_sha, qwen_params
            )
            print(json.dumps(status, indent=2))
            return

        utils.check_ollama(qwen_params["model"])
        train = frame[frame["set"].eq("train")].copy().reset_index(drop=True)
        summaries = []
        configurations = qwen_inference.experiment_configurations(qwen_params)
        for profile, shots in configurations:
            dataset_name = qwen_inference.CACHE_DATASETS[(profile, shots)]
            # Un smoke test nunca lee ni sobrescribe los caches canónicos.
            existing = pd.DataFrame() if args.limit is not None else caches[(profile, shots)]
            save_cache = (
                None
                if args.limit is not None
                else lambda value, name=dataset_name: catalog.save(name, value)
            )
            summaries.append(qwen_inference.run_configuration(
                frame=frame,
                train=train,
                features=features[profile],
                dataset_sha=dataset_sha,
                profile=profile,
                shots=shots,
                params=qwen_params,
                existing=existing,
                save_cache=save_cache,
                limit=args.limit,
            ))

        if args.limit is None:
            catalog.save(qwen_inference.SUMMARY_DATASET, summaries)
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()

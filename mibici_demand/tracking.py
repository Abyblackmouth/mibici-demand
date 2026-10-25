"""Versionado de datasets con MLflow: cada versión queda como un run con su dataset, sus reglas y sus métricas."""

from __future__ import annotations

from pathlib import Path
import subprocess

import mlflow
import pandas as pd

from mibici_demand import config


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "sin-commit"


def log_dataset_version(
    name: str,
    version: str,
    dataset_df: pd.DataFrame,
    source: str | Path,
    *,
    context: str,
    params: dict,
    metrics: dict,
    artifacts: list[Path] | None = None,
    tags: dict | None = None,
) -> str:
    """Registra una versión de un dataset: el dataset como entrada del run (con su huella digital), los parámetros que lo
    definen, las métricas que lo describen y los archivos de documentación. Devuelve el identificador del run."""
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{name}-{version}") as run:
        dataset = mlflow.data.from_pandas(dataset_df, source=str(source), name=name)
        mlflow.log_input(dataset, context=context)
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.set_tags(
            {"dataset": name, "version": version, "git_commit": _git_commit(), **(tags or {})}
        )
        for path in artifacts or []:
            mlflow.log_artifact(str(path), artifact_path="documentacion")
        return run.info.run_id

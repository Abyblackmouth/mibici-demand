"""Versionado de datasets y seguimiento/registro de modelos con MLflow (Fases 1 y 2)."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
from typing import Any

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
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


def _model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "baseline" in name_lower or "168h" in name_lower or "24h" in name_lower:
        return "baseline_estacional"
    if model_name in {"hist_gbm_poisson", "random_forest"}:
        return "ensamble_arboles"
    if model_name in {"ridge", "poisson"}:
        return "modelo_lineal_glm"
    return "otro"


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


# ---------------------------------------------------------------------------------------------------------------
# Fase 2: Seguimiento de Experimentos (MLflow Tracking) y Gestión en MLflow Model Registry
# ---------------------------------------------------------------------------------------------------------------
def log_model_run(
    model_name: str,
    metrics: dict[str, float],
    params: dict[str, Any] | None = None,
    *,
    horizon: str = config.DEFAULT_HORIZON,
    etiqueta_ventana: str = "completa",
    max_train_months: int | None = 12,
    filas_entrenamiento: int = 0,
    filas_prueba: int = 0,
    sha256_tabla: str = "",
    baseline_wape: float | None = None,
    pipeline: Any | None = None,
    X_sample: pd.DataFrame | None = None,
    cv_trials: Sequence[dict[str, Any]] | None = None,
    artifacts: Sequence[Path | str] | None = None,
    tags: dict[str, str] | None = None,
    nested: bool = False,
    artifact_path: str = "model",
) -> str:
    """Registra la ejecución de un baseline o modelo de ML en MLflow Tracking (`config.MODEL_EXPERIMENT_NAME`).

    Soporta runs anidados (`nested=True`) para los ensayos de búsqueda de hiperparámetros (`cv_trials`)
    y empaqueta el `Pipeline` de Scikit-Learn junto con su `signature` e `input_example`.

    Parameters
    ----------
    model_name : str
        Nombre del modelo o baseline (`"baseline_estacional_168h"`, `"ridge"`, `"poisson"`,
        `"hist_gbm_poisson"`, `"random_forest"`).
    metrics : dict[str, float]
        Métricas evaluadas (`mae`, `rmse`, `wape`, `r2`, `mae_horas_pico`, `wape_horas_pico`, etc.).
    params : dict[str, Any] | None, optional
        Mejores hiperparámetros del algoritmo o configuración del baseline.
    horizon : str, default=config.DEFAULT_HORIZON
        Horizonte operativo (`"24h"` o `"1h"`).
    etiqueta_ventana : str, default="completa"
        Identificador de la ventana histórica (`"completa"` o `"desde2022"`).
    max_train_months : int | None, default=12
        Meses máximos empleados en el entrenamiento.
    filas_entrenamiento : int, default=0
        Número de filas en el conjunto de entrenamiento.
    filas_prueba : int, default=0
        Número de filas en el conjunto de prueba.
    sha256_tabla : str, default=""
        Huella SHA-256 de la tabla procesada utilizada.
    baseline_wape : float | None, optional
        WAPE del baseline de referencia para calcular `mejora_wape_vs_baseline_pct`.
    pipeline : Any | None, optional
        Pipeline de Scikit-Learn entrenado para registrarse con `mlflow.sklearn.log_model`.
    X_sample : pd.DataFrame | None, optional
        Muestra de entrada para inferir la firma (`infer_signature`) y guardar `input_example`.
    cv_trials : Sequence[dict[str, Any]] | None, optional
        Ensayos de validación cruzada para registrarse como runs hijos anidados (`nested=True`).
    artifacts : Sequence[Path | str] | None, optional
        Rutas de figuras o reportes CSV/JSON a adjuntar al run.
    tags : dict[str, str] | None, optional
        Etiquetas adicionales para el run.
    nested : bool, default=False
        Indica si este run se abre como anidado dentro de un run padre activo.
    artifact_path : str, default="model"
        Subcarpeta del artefacto del modelo dentro del run de MLflow.

    Returns
    -------
    str
        Identificador `run_id` generado en MLflow.
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.MODEL_EXPERIMENT_NAME)

    run_params: dict[str, str] = {
        "algoritmo": str(model_name),
        "horizonte": str(horizon),
        "etiqueta_ventana": str(etiqueta_ventana),
        "max_train_months": str(max_train_months if max_train_months is not None else "todos"),
        "filas_entrenamiento": str(filas_entrenamiento),
        "filas_prueba": str(filas_prueba),
        "sha256_tabla": str(sha256_tabla or "no-especificado"),
    }
    for k, v in (params or {}).items():
        run_params[str(k)] = str(v)

    run_metrics: dict[str, float] = {
        k: float(v) for k, v in metrics.items() if v is not None and np.isfinite(float(v))
    }
    if "mejora_wape_vs_baseline_pct" not in run_metrics:
        current_wape = run_metrics.get("wape", run_metrics.get("test_wape"))
        if baseline_wape is not None and baseline_wape > 0.0 and current_wape is not None:
            run_metrics["mejora_wape_vs_baseline_pct"] = float(
                (baseline_wape - current_wape) / baseline_wape * 100.0
            )
        else:
            run_metrics["mejora_wape_vs_baseline_pct"] = 0.0

    familia = _model_family(model_name)
    run_tags: dict[str, str] = {
        "stage": "fase-2",
        "git_commit": _git_commit(),
        "familia_modelo": familia,
        "horizonte": str(horizon),
        "ventana": str(etiqueta_ventana),
        **(tags or {}),
    }

    with mlflow.start_run(
        run_name=f"{model_name}-{horizon}-{etiqueta_ventana}-{max_train_months or 'todos'}m",
        nested=nested,
    ) as run:
        mlflow.log_params(run_params)
        mlflow.log_metrics(run_metrics)
        mlflow.set_tags(run_tags)

        # Registro de ensayos de búsqueda de hiperparámetros como runs hijos anidados
        for idx, trial in enumerate(cv_trials or [], start=1):
            trial_params = trial.get("params", {}) if isinstance(trial, dict) else {}
            trial_metrics = trial.get("metrics", {}) if isinstance(trial, dict) else {}
            with mlflow.start_run(
                run_name=f"{model_name}-trial-{idx}",
                nested=True,
            ):
                mlflow.log_params(
                    {
                        "algoritmo": str(model_name),
                        "horizonte": str(horizon),
                        "trial_index": str(idx),
                        **{str(k): str(v) for k, v in trial_params.items()},
                    }
                )
                if trial_metrics:
                    mlflow.log_metrics(
                        {
                            str(k): float(v)
                            for k, v in trial_metrics.items()
                            if v is not None and np.isfinite(float(v))
                        }
                    )
                mlflow.set_tags(
                    {
                        "stage": "fase-2",
                        "git_commit": run_tags["git_commit"],
                        "familia_modelo": familia,
                        "horizonte": str(horizon),
                        "ventana": str(etiqueta_ventana),
                        "tipo_run": "tuning_trial",
                    }
                )

        if pipeline is not None:
            signature = None
            input_example = None
            if X_sample is not None and len(X_sample) > 0:
                sample_df = X_sample.head(min(len(X_sample), 20)).copy()
                y_pred_sample = np.clip(
                    np.asarray(pipeline.predict(sample_df), dtype=np.float64), 0.0, None
                )
                signature = infer_signature(sample_df, y_pred_sample)
                input_example = sample_df.head(min(len(sample_df), 5))

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

        for path in artifacts or []:
            artifact_file = Path(path)
            if artifact_file.exists():
                mlflow.log_artifact(str(artifact_file), artifact_path="reportes")

        return run.info.run_id


def register_champion_model(
    run_id: str | None = None,
    model_name: str = "hist_gbm_poisson",
    metrics: dict[str, float] | None = None,
    best_params: dict[str, Any] | None = None,
    *,
    etiqueta_ventana: str = "completa",
    horizon: str = config.DEFAULT_HORIZON,
    registered_model_name: str = config.REGISTERED_MODEL_NAME,
    artifact_path: str = "model",
    alias: str = "champion",
    pipeline: Any | None = None,
    X_sample: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Registra el mejor modelo en MLflow Model Registry, documenta su versión y asigna el alias `'champion'`.

    Parameters
    ----------
    run_id : str | None, optional
        ID del run de MLflow que contiene el artefacto del modelo. Si es `None` y se proporciona
        `pipeline`, se crea primero el run con `log_model_run`.
    model_name : str, default="hist_gbm_poisson"
        Nombre del algoritmo ganador.
    metrics : dict[str, float] | None, optional
        Diccionario con las métricas de evaluación (`mae`, `rmse`, `wape`, `wape_horas_pico`).
    best_params : dict[str, Any] | None, optional
        Hiperparámetros óptimos seleccionados para el modelo campeón.
    etiqueta_ventana : str, default="completa"
        Ventana de datos utilizada (`"completa"` o `"desde2022"`).
    horizon : str, default=config.DEFAULT_HORIZON
        Horizonte operativo de predicción (`"24h"` o `"1h"`).
    registered_model_name : str, default=config.REGISTERED_MODEL_NAME
        Nombre del modelo en el Model Registry (`"mibici-demanda-estacion-hora"`).
    artifact_path : str, default="model"
        Ruta del artefacto del modelo dentro del run.
    alias : str, default="champion"
        Alias asignado a la versión promovida en el registro.
    pipeline : Any | None, optional
        Pipeline de Scikit-Learn para registrar si `run_id` no fue provisto.
    X_sample : pd.DataFrame | None, optional
        Muestra de entrada para inferir firma si se registra `pipeline` directamente.

    Returns
    -------
    dict[str, Any]
        Metadatos de la versión registrada (`name`, `version`, `alias`, `run_id`, `model_uri`, `description`).
    """
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.MODEL_EXPERIMENT_NAME)

    metrics_dict = dict(metrics or {})
    params_dict = dict(best_params or {})

    if run_id is None:
        if pipeline is None:
            raise ValueError(
                "Debe proporcionar `run_id` de un run con modelo o `pipeline` para registrarlo."
            )
        run_id = log_model_run(
            model_name=model_name,
            metrics=metrics_dict,
            params=params_dict,
            horizon=horizon,
            etiqueta_ventana=etiqueta_ventana,
            pipeline=pipeline,
            X_sample=X_sample,
            artifact_path=artifact_path,
        )

    model_uri = f"runs:/{run_id}/{artifact_path}"
    model_version = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
    version_str = str(model_version.version)

    mae_val = metrics_dict.get("mae", metrics_dict.get("test_mae", 0.0))
    rmse_val = metrics_dict.get("rmse", metrics_dict.get("test_rmse", 0.0))
    wape_val = metrics_dict.get("wape", metrics_dict.get("test_wape", 0.0))
    wape_peak_val = metrics_dict.get(
        "wape_horas_pico", metrics_dict.get("test_wape_horas_pico", 0.0)
    )
    params_json = json.dumps(params_dict, ensure_ascii=False)

    description = (
        f"Versión v{version_str} del modelo de pronóstico de demanda horaria MiBici Guadalajara "
        f"(horizonte={horizon}, ventana={etiqueta_ventana}). "
        f"Algoritmo ganador: {model_name} | Hiperparámetros óptimos: {params_json} | "
        f"Métricas en prueba -> MAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}, "
        f"WAPE: {wape_val:.4f}, WAPE horas pico: {wape_peak_val:.4f}."
    )

    client = MlflowClient(tracking_uri=config.TRACKING_URI)
    client.update_model_version(
        name=registered_model_name,
        version=version_str,
        description=description,
    )

    version_tags = {
        "version": version_str,
        "algoritmo": str(model_name),
        "hiperparametros": params_json,
        "mae": f"{mae_val:.4f}",
        "rmse": f"{rmse_val:.4f}",
        "wape": f"{wape_val:.4f}",
        "wape_horas_pico": f"{wape_peak_val:.4f}",
        "ventana": str(etiqueta_ventana),
        "horizonte": str(horizon),
        "stage": "fase-2",
    }
    for key, value in version_tags.items():
        client.set_model_version_tag(
            name=registered_model_name,
            version=version_str,
            key=key,
            value=value,
        )

    client.set_registered_model_alias(
        name=registered_model_name,
        alias=alias,
        version=version_str,
    )

    return {
        "name": registered_model_name,
        "version": version_str,
        "alias": alias,
        "run_id": run_id,
        "model_uri": f"models:/{registered_model_name}@{alias}",
        "description": description,
    }

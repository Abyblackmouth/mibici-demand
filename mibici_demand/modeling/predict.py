"""Inferencia por lotes para pronóstico de demanda horaria y priorización de redistribución en MiBici.

Este módulo carga el pipeline entrenado (desde `models/best_pipeline.joblib` o desde un URI
del Model Registry de MLflow como `models:/mibici-demanda-estacion-hora@champion`),
genera la columna `demanda_predicha` (truncada a valores no negativos `>= 0.0`) y, cuando
la capacidad de anclajes (`capacidad`) está disponible en la tabla de entrada, calcula el
indicador operativo `presion_redistribucion = demanda_predicha / capacidad` para priorizar
las estaciones con mayor riesgo de vaciamiento o saturación en las próximas 24 horas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici_demand import config, features
from mibici_demand.modeling.train import BEST_PIPELINE_FILENAME, DEFAULT_DATASET_FILENAME

app = typer.Typer(help="Inferencia por lotes para pronóstico de demanda por estación y hora.")

DEFAULT_PREDICTIONS_FILENAME = "predicciones_demanda.parquet"
PREDICTED_COLUMN = "demanda_predicha"
REDISTRIBUTION_PRESSURE_COLUMN = "presion_redistribucion"
MLFLOW_URI_PREFIXES = ("models:/", "runs:/")


def load_pipeline(model_source: Path | str = config.MODELS_DIR / BEST_PIPELINE_FILENAME) -> Any:
    """Carga el pipeline de modelado desde un archivo local `.joblib` o desde MLflow Model Registry.

    Parameters
    ----------
    model_source : Path | str, default=config.MODELS_DIR / "best_pipeline.joblib"
        Ruta local al artefacto serializado con `joblib` o URI de MLflow (`models:/...` o `runs:/...`).

    Returns
    -------
    Any
        Pipeline de Scikit-Learn o modelo pyfunc/sklearn cargado desde MLflow listo para `.predict(X)`.
    """
    source_str = str(model_source)
    if source_str.startswith(MLFLOW_URI_PREFIXES):
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(config.TRACKING_URI)
        logger.info(f"Cargando modelo desde MLflow Registry: {source_str}")
        return mlflow.sklearn.load_model(source_str)

    model_path = Path(model_source)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el pipeline entrenado en {model_path}. "
            "Ejecute primero `python -m mibici_demand.modeling.train`."
        )
    logger.info(f"Cargando pipeline local desde {model_path}")
    return joblib.load(model_path)


def predict_demand(
    df: pd.DataFrame,
    pipeline: Any,
    horizon: str = config.DEFAULT_HORIZON,
) -> pd.DataFrame:
    """Genera las predicciones de demanda horaria y el índice de presión de redistribución.

    Parameters
    ----------
    df : pd.DataFrame
        Tabla con las características de entrada por estación y hora.
    pipeline : Any
        Pipeline entrenado que implementa `.predict(X)`.
    horizon : str, default=config.DEFAULT_HORIZON
        Horizonte operativo (`"24h"` o `"1h"`) para seleccionar las columnas predictoras válidas.

    Returns
    -------
    pd.DataFrame
        Copia del `DataFrame` de entrada enriquecida con `demanda_predicha` (`>= 0.0`) y,
        si existe la columna `capacidad`, con `presion_redistribucion = demanda_predicha / capacidad`.
    """
    if df.empty:
        raise ValueError("El DataFrame de entrada para inferencia está vacío.")

    feature_cols = features.get_feature_columns(df, horizon=horizon)
    raw_predictions = np.asarray(pipeline.predict(df[feature_cols]), dtype=np.float64)
    clipped_predictions = np.clip(raw_predictions, 0.0, None).astype("float32")

    out = df.copy()
    out[PREDICTED_COLUMN] = clipped_predictions

    if "capacidad" in out.columns:
        capacidad = pd.to_numeric(out["capacidad"], errors="coerce").to_numpy(dtype=np.float64)
        valid_cap = np.isfinite(capacidad) & (capacidad > 0.0)
        pressure = np.where(
            valid_cap,
            clipped_predictions.astype(np.float64) / np.where(valid_cap, capacidad, 1.0),
            np.nan,
        )
        out[REDISTRIBUTION_PRESSURE_COLUMN] = pressure.astype("float32")

    return out


def run_batch_inference(
    features_path: Path = config.PROCESSED_DATA_DIR / DEFAULT_DATASET_FILENAME,
    model_source: Path | str = config.MODELS_DIR / BEST_PIPELINE_FILENAME,
    predictions_path: Path = config.PROCESSED_DATA_DIR / DEFAULT_PREDICTIONS_FILENAME,
    horizon: str = config.DEFAULT_HORIZON,
    last_days: int | None = config.TEST_DAYS,
) -> pd.DataFrame:
    """Ejecuta el flujo de inferencia por lotes desde disco y exporta las predicciones.

    Parameters
    ----------
    features_path : Path
        Ruta de la tabla de entrada (`.parquet` o `.csv`).
    model_source : Path | str
        Ruta local a `models/best_pipeline.joblib` o URI `models:/...` de MLflow.
    predictions_path : Path
        Ruta de salida donde se guardan las predicciones (`data/processed/predicciones_demanda.parquet`).
    horizon : str, default=config.DEFAULT_HORIZON
        Horizonte operativo de predicción (`"24h"` o `"1h"`).
    last_days : int | None, default=config.TEST_DAYS
        Si se especifica y la tabla contiene `hora_inicio`, filtra los últimos `last_days` días
        para ejecutar la inferencia operativa sin cargar innecesariamente toda la década en salida.

    Returns
    -------
    pd.DataFrame
        Tabla con las predicciones generadas.
    """
    if not features_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada en {features_path}.")

    if features_path.suffix.lower() == ".csv":
        df = pd.read_csv(features_path)
        if "hora_inicio" in df.columns:
            df["hora_inicio"] = pd.to_datetime(df["hora_inicio"])
    else:
        parquet_filters = None
        if last_days is not None and last_days > 0:
            summary_candidate = features_path.with_name(
                features_path.stem.replace("demanda_estacion_hora_", "resumen_tabla_") + ".json"
            )
            if summary_candidate.exists():
                import json

                info = json.loads(summary_candidate.read_text(encoding="utf-8"))
                if info.get("hora_max"):
                    min_ts = pd.Timestamp(info["hora_max"]).floor("D") - pd.Timedelta(
                        days=last_days - 1
                    )
                    parquet_filters = [("hora_inicio", ">=", min_ts)]
        df = pd.read_parquet(features_path, filters=parquet_filters)

    if last_days is not None and last_days > 0 and "hora_inicio" in df.columns:
        cutoff = df["hora_inicio"].max().floor("D") - pd.Timedelta(days=last_days - 1)
        df = df[df["hora_inicio"] >= cutoff].reset_index(drop=True)

    pipeline = load_pipeline(model_source)
    result = predict_demand(df, pipeline=pipeline, horizon=horizon)

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.suffix.lower() == ".csv":
        result.to_csv(predictions_path, index=False)
    else:
        result.to_parquet(predictions_path, index=False)

    logger.success(
        f"Inferencia completada ({len(result):,} filas) -> guardada en {predictions_path}"
    )
    return result


@app.command()
def main(
    features_path: Path = config.PROCESSED_DATA_DIR / DEFAULT_DATASET_FILENAME,
    model_path: str = str(config.MODELS_DIR / BEST_PIPELINE_FILENAME),
    predictions_path: Path = config.PROCESSED_DATA_DIR / DEFAULT_PREDICTIONS_FILENAME,
    horizon: str = config.DEFAULT_HORIZON,
    last_days: int = config.TEST_DAYS,
) -> None:
    """Comando CLI para generar predicciones por lotes y presión de redistribución."""
    run_batch_inference(
        features_path=features_path,
        model_source=model_path,
        predictions_path=predictions_path,
        horizon=horizon,
        last_days=last_days,
    )


if __name__ == "__main__":
    app()

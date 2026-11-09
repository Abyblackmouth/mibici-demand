"""Entrenamiento, ajuste de hiperparámetros y evaluación de modelos para MiBici Guadalajara (Fase 2).

Interacción de los 5 roles de MLOps en esta fase
------------------------------------------------
1. **Data Engineer (Ingeniería de Datos)**:
   Entrega la tabla histórica limpia y agregada por estación-hora
   (`data/processed/demanda_estacion_hora_limpio.parquet` y su variante `desde2022`),
   garantizando el esquema de columnas, la exclusión de horas sin servicio, el cálculo
   de rezagos sin fuga temporal y la integración del catálogo geográfico (`latitud`,
   `longitud`, `capacidad`).
2. **Data Scientist (Ciencia de Datos)**:
   Formula la predicción de viajes iniciados como un problema de regresión de conteos no
   negativos con exceso de ceros (43.4%, H12 en `references/hallazgos_de_datos.md`),
   selecciona cuatro familias de algoritmos (`ridge`, `poisson`, `hist_gbm_poisson`,
   `random_forest`), restringe los predictores al horizonte operativo de 24 horas
   (`lag_24`, `lag_168`) y define métricas globales (`mae`, `rmse`, `wape`, `r2`) junto
   con métricas operativas en horas pico (`mae_horas_pico`, `wape_horas_pico` en
   `config.PEAK_HOURS`) comparadas contra los dos baselines de la Fase 1
   (`estacional_168h` y `ayer_24h`).
3. **ML Engineer (Ingeniería de Machine Learning)**:
   Construye el `Pipeline` extremo a extremo (`build_model_pipeline`), ejecuta la búsqueda
   de hiperparámetros con validación cruzada temporal (`TimeSeriesSplit` +
   `RandomizedSearchCV`), reentrena cada mejor configuración sobre `train + val`, evalúa
   en `test`, persiste el mejor pipeline en `models/best_pipeline.joblib` y registra
   experimentos en MLflow (`config.MODEL_EXPERIMENT_NAME`).
4. **Software Engineer (Ingeniería de Software)**:
   Asegura un diseño modular, tipado estático, parámetros centralizados en
   `mibici_demand.config` (sin constantes mágicas), interfaz de línea de comandos con
   `typer` y pruebas automatizadas reproducibles con `pytest`.
5. **SRE / Operaciones MLOps (Confiabilidad y Recursos)**:
   Protege el límite operativo de 8 GB de RAM acotando la ventana histórica de
   entrenamiento (`max_train_months`), evitando copias paralelas innecesarias durante la
   validación cruzada, liberando memoria intermedia (`gc.collect()`) y fijando semillas
   (`config.RANDOM_SEED`) para reproducibilidad determinista.
"""

from __future__ import annotations

from collections.abc import Sequence
import gc
import json
from pathlib import Path
import time
from typing import Any

import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import typer

from mibici_demand import config, features, plots, tracking
from mibici_demand.modeling.pipeline import SUPPORTED_MODELS, build_model_pipeline

app = typer.Typer(help="Entrenamiento, ajuste temporal y evaluación de modelos de demanda MiBici.")

# Constantes configurables del proceso de ajuste y evaluación (sin números mágicos)
DEFAULT_DATASET_FILENAME = "demanda_estacion_hora_limpio.parquet"
BEST_PIPELINE_FILENAME = "best_pipeline.joblib"
RESULTS_CSV_FILENAME = "resultados_modelos.csv"
RESULTS_JSON_FILENAME = "resultados_modelos.json"

DEFAULT_MAX_TRAIN_MONTHS = 12
DEFAULT_CV_SPLITS = 3
DEFAULT_N_ITER = 4
DEFAULT_MAX_TUNING_ROWS = 300_000
RANDOM_FOREST_MAX_FIT_ROWS = 250_000
DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    "ridge",
    "poisson",
    "hist_gbm_poisson",
    "random_forest",
)

# Espacio de búsqueda de hiperparámetros para cada familia de modelos (prefijo `modelo__` del Pipeline)
RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
POISSON_ALPHAS = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
POISSON_MAX_ITERS = [200, 300, 500]
HGBR_LEARNING_RATES = [0.03, 0.05, 0.1]
HGBR_MAX_ITERS = [150, 250, 350]
HGBR_MAX_LEAF_NODES = [31, 63, 127]
HGBR_MIN_SAMPLES_LEAF = [20, 50, 100]
HGBR_L2_REGULARIZATION = [0.0, 0.1, 1.0]
RF_N_ESTIMATORS = [50, 100, 150]
RF_MAX_DEPTHS = [10, 15, 20]
RF_MIN_SAMPLES_LEAF = [5, 10, 20]
RF_MAX_FEATURES = ["sqrt", 0.7]

PARAM_DISTRIBUTIONS: dict[str, dict[str, list[Any]]] = {
    "ridge": {
        "modelo__alpha": RIDGE_ALPHAS,
    },
    "poisson": {
        "modelo__alpha": POISSON_ALPHAS,
        "modelo__max_iter": POISSON_MAX_ITERS,
    },
    "hist_gbm_poisson": {
        "modelo__learning_rate": HGBR_LEARNING_RATES,
        "modelo__max_iter": HGBR_MAX_ITERS,
        "modelo__max_leaf_nodes": HGBR_MAX_LEAF_NODES,
        "modelo__min_samples_leaf": HGBR_MIN_SAMPLES_LEAF,
        "modelo__l2_regularization": HGBR_L2_REGULARIZATION,
    },
    "random_forest": {
        "modelo__n_estimators": RF_N_ESTIMATORS,
        "modelo__max_depth": RF_MAX_DEPTHS,
        "modelo__min_samples_leaf": RF_MIN_SAMPLES_LEAF,
        "modelo__max_features": RF_MAX_FEATURES,
    },
}


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    hours: np.ndarray | pd.Series | None = None,
) -> dict[str, float]:
    """Calcula métricas de evaluación globales y en horas pico para conteos de demanda.

    Las predicciones se recortan a valores no negativos (`>= 0.0`) dado que la demanda
    de viajes iniciados por estación y hora no puede ser negativa.

    Parameters
    ----------
    y_true : np.ndarray | pd.Series
        Valores reales observados de `demanda`.
    y_pred : np.ndarray | pd.Series
        Valores predichos por el modelo o baseline.
    hours : np.ndarray | pd.Series | None, optional
        Hora del día (`0..23`) asociada a cada observación. Si se proporciona, calcula
        adicionalmente `mae_horas_pico` y `wape_horas_pico` sobre `config.PEAK_HOURS`.

    Returns
    -------
    dict[str, float]
        Diccionario con `mae`, `rmse`, `wape`, `r2` y, cuando `hours` está presente,
        `mae_horas_pico` y `wape_horas_pico`.
    """
    y_obs = np.asarray(y_true, dtype=np.float64)
    y_hat = np.clip(np.asarray(y_pred, dtype=np.float64), 0.0, None)

    abs_err = np.abs(y_obs - y_hat)
    total_obs = float(y_obs.sum())
    wape = float(abs_err.sum() / total_obs) if total_obs > 0.0 else 0.0
    r2 = float(r2_score(y_obs, y_hat)) if len(y_obs) >= 2 else 0.0

    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_obs, y_hat)),
        "rmse": float(np.sqrt(mean_squared_error(y_obs, y_hat))),
        "wape": wape,
        "r2": r2,
    }

    if hours is not None:
        hours_arr = np.asarray(hours)
        peak_mask = np.isin(hours_arr, config.PEAK_HOURS)
        if np.any(peak_mask):
            y_obs_peak = y_obs[peak_mask]
            y_hat_peak = y_hat[peak_mask]
            peak_err = np.abs(y_obs_peak - y_hat_peak)
            peak_total = float(y_obs_peak.sum())
            metrics["mae_horas_pico"] = float(peak_err.mean())
            metrics["wape_horas_pico"] = (
                float(peak_err.sum() / peak_total) if peak_total > 0.0 else 0.0
            )
        else:
            metrics["mae_horas_pico"] = 0.0
            metrics["wape_horas_pico"] = 0.0

    return metrics


def temporal_train_val_test_split(
    table: pd.DataFrame,
    val_days: int = config.VAL_DAYS,
    test_days: int = config.TEST_DAYS,
    max_train_months: int | None = DEFAULT_MAX_TRAIN_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide la tabla cronológicamente en Entrenamiento, Validación y Prueba sin fuga temporal.

    Garantiza que:
    `train["hora_inicio"].max() < val["hora_inicio"].min() < test["hora_inicio"].min()`.
    Opcionalmente acota el conjunto de entrenamiento a los últimos `max_train_months`
    meses previos al corte de validación para mantener el consumo de memoria bajo 8 GB RAM.

    Parameters
    ----------
    table : pd.DataFrame
        Tabla estación-hora con columna `hora_inicio`.
    val_days : int, default=config.VAL_DAYS
        Número de días previos al conjunto de prueba reservados para validación.
    test_days : int, default=config.TEST_DAYS
        Número de días finales reservados para la evaluación de prueba (consistente con Fase 1).
    max_train_months : int | None, default=12
        Cantidad máxima de meses de historia en `train`. Si es `None`, conserva toda la historia.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tupla `(train_df, val_df, test_df)` ordenada cronológicamente por `hora_inicio` y `estacion`.
    """
    if "hora_inicio" not in table.columns:
        raise ValueError("La tabla debe incluir la columna temporal 'hora_inicio'.")

    ordered = table.sort_values(["hora_inicio", "estacion"]).reset_index(drop=True)
    max_day = ordered["hora_inicio"].max().floor("D")
    test_cutoff = max_day - pd.Timedelta(days=test_days - 1)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)

    train_mask = ordered["hora_inicio"] < val_cutoff
    if max_train_months is not None and max_train_months > 0:
        train_start = val_cutoff - pd.DateOffset(months=max_train_months)
        bounded_mask = train_mask & (ordered["hora_inicio"] >= train_start)
        if bounded_mask.any():
            train_mask = bounded_mask

    val_mask = (ordered["hora_inicio"] >= val_cutoff) & (ordered["hora_inicio"] < test_cutoff)
    test_mask = ordered["hora_inicio"] >= test_cutoff

    train_df = ordered.loc[train_mask].reset_index(drop=True)
    val_df = ordered.loc[val_mask].reset_index(drop=True)
    test_df = ordered.loc[test_mask].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "La partición temporal produjo un subconjunto vacío "
            f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)}). "
            "Verifique el rango de fechas de la tabla o reduzca `val_days` / `test_days`."
        )

    return train_df, val_df, test_df


def _evaluate_baselines(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Evalúa los dos baselines de referencia de la Fase 1 en validación y prueba."""
    rows: list[dict[str, Any]] = []
    for name, lag_col in (
        ("baseline_estacional_168h", "lag_168"),
        ("baseline_ayer_24h", "lag_24"),
    ):
        if lag_col not in test_df.columns:
            continue
        val_metrics = compute_metrics(
            val_df[features.TARGET], val_df[lag_col], hours=val_df["hora"]
        )
        test_metrics = compute_metrics(
            test_df[features.TARGET], test_df[lag_col], hours=test_df["hora"]
        )
        rows.append(
            {
                "modelo": name,
                "tipo": "baseline",
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
                "val_wape": val_metrics["wape"],
                "val_r2": val_metrics["r2"],
                "val_mae_horas_pico": val_metrics["mae_horas_pico"],
                "val_wape_horas_pico": val_metrics["wape_horas_pico"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_wape": test_metrics["wape"],
                "test_r2": test_metrics["r2"],
                "test_mae_horas_pico": test_metrics["mae_horas_pico"],
                "test_wape_horas_pico": test_metrics["wape_horas_pico"],
                "tiempo_seg": 0.0,
                "mejores_parametros": json.dumps({"columna_referencia": lag_col}),
            }
        )
    return rows


def _strip_model_prefix(best_params: dict[str, Any]) -> dict[str, Any]:
    """Elimina el prefijo `modelo__` devuelto por `RandomizedSearchCV` sobre el `Pipeline`."""
    prefix = "modelo__"
    return {k.removeprefix(prefix): v for k, v in best_params.items()}


def _total_combinations(param_grid: dict[str, list[Any]]) -> int:
    total = 1
    for values in param_grid.values():
        total *= max(1, len(values))
    return total


def tune_and_evaluate(
    table: pd.DataFrame,
    models: Sequence[str] = DEFAULT_MODEL_NAMES,
    horizon: str = config.DEFAULT_HORIZON,
    include_station_meta: bool = True,
    val_days: int = config.VAL_DAYS,
    test_days: int = config.TEST_DAYS,
    max_train_months: int | None = DEFAULT_MAX_TRAIN_MONTHS,
    n_iter: int = DEFAULT_N_ITER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    max_tuning_rows: int | None = DEFAULT_MAX_TUNING_ROWS,
    models_dir: Path = config.MODELS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    log_to_mlflow: bool = False,
    etiqueta_ventana: str = "completa",
    sha256_tabla: str = "",
) -> dict[str, Any]:
    """Ajusta hiperparámetros con validación temporal, reentrena en `train + val` y evalúa en `test`.

    Para cada modelo en `models`:
    1. Construye el `Pipeline` completo mediante `build_model_pipeline`.
    2. Ejecuta `RandomizedSearchCV` con `TimeSeriesSplit(n_splits=cv_splits)` sobre el subconjunto
       de entrenamiento (acotado a las últimas `max_tuning_rows` filas si se especifica para
       respetar el límite de 8 GB de RAM).
    3. Evalúa el mejor estimador de búsqueda en `val_df`.
    4. Reentrena un `Pipeline` limpio con los mejores hiperparámetros usando `train_df + val_df`
       y evalúa su desempeño final en `test_df` frente a los dos baselines (`estacional_168h`
       y `ayer_24h`).
    5. Persiste el mejor pipeline en `models_dir / "best_pipeline.joblib"` y exporta la tabla
       comparativa en `reports_dir / "resultados_modelos.csv"` y `.json`.

    Parameters
    ----------
    table : pd.DataFrame
        Tabla de características y objetivo (`demanda`) por estación y hora.
    models : Sequence[str], default=DEFAULT_MODEL_NAMES
        Lista de nombres de algoritmos a comparar (`"ridge"`, `"poisson"`,
        `"hist_gbm_poisson"`, `"random_forest"`).
    horizon : str, default=config.DEFAULT_HORIZON
        Horizonte operativo (`"24h"` o `"1h"`).
    include_station_meta : bool, default=True
        Si es `True`, incluye `latitud`, `longitud` y `capacidad` cuando estén disponibles.
    val_days : int, default=config.VAL_DAYS
        Días del subconjunto de validación.
    test_days : int, default=config.TEST_DAYS
        Días del subconjunto de prueba final.
    max_train_months : int | None, default=DEFAULT_MAX_TRAIN_MONTHS
        Meses máximos de historia de entrenamiento.
    n_iter : int, default=DEFAULT_N_ITER
        Iteraciones de búsqueda aleatoria en `RandomizedSearchCV`.
    cv_splits : int, default=DEFAULT_CV_SPLITS
        Número de particiones en `TimeSeriesSplit`.
    max_tuning_rows : int | None, default=DEFAULT_MAX_TUNING_ROWS
        Límite superior de filas cronológicas más recientes para la búsqueda de hiperparámetros.
    models_dir : Path, default=config.MODELS_DIR
        Directorio donde se guarda `best_pipeline.joblib`.
    reports_dir : Path, default=config.REPORTS_DIR
        Directorio donde se exportan `resultados_modelos.csv` y `resultados_modelos.json`.
    log_to_mlflow : bool, default=False
        Si es `True`, registra ejecuciones y métricas en MLflow bajo `config.MODEL_EXPERIMENT_NAME`.

    Returns
    -------
    dict[str, Any]
        Diccionario con el modelo ganador, su pipeline entrenado, rutas de salida y `DataFrame`
        comparativo de resultados.
    """
    invalid_models = [m for m in models if m not in SUPPORTED_MODELS]
    if invalid_models:
        raise ValueError(
            f"Modelos no soportados: {invalid_models}. Opciones válidas: {sorted(SUPPORTED_MODELS)}"
        )

    train_df, val_df, test_df = temporal_train_val_test_split(
        table,
        val_days=val_days,
        test_days=test_days,
        max_train_months=max_train_months,
    )

    feature_cols = features.get_feature_columns(train_df, horizon=horizon)
    if not include_station_meta:
        feature_cols = [c for c in feature_cols if c not in config.STATION_META_COLS]

    logger.info(
        f"Partición temporal lista | train={len(train_df):,} | val={len(val_df):,} | "
        f"test={len(test_df):,} | columnas={feature_cols}"
    )

    results_rows: list[dict[str, Any]] = _evaluate_baselines(val_df=val_df, test_df=test_df)

    # Subconjunto cronológico más reciente para tuning eficiente dentro de 8 GB RAM
    if max_tuning_rows is not None and len(train_df) > max_tuning_rows:
        tune_df = train_df.iloc[-max_tuning_rows:].reset_index(drop=True)
    else:
        tune_df = train_df

    X_tune = tune_df[feature_cols]
    y_tune = tune_df[features.TARGET].to_numpy(np.float64)
    X_val = val_df[feature_cols]
    y_val = val_df[features.TARGET].to_numpy(np.float64)
    X_test = test_df[feature_cols]
    y_test = test_df[features.TARGET].to_numpy(np.float64)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    best_model_name: str | None = None
    best_pipeline = None
    best_test_mae = float("inf")
    best_val_mae = float("inf")
    best_test_pred: np.ndarray | None = None
    best_model_params: dict[str, Any] = {}
    best_model_test_metrics: dict[str, float] = {}
    model_artifacts_data: list[dict[str, Any]] = []

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    for model_name in models:
        t0 = time.perf_counter()
        logger.info(
            f"Ajustando modelo '{model_name}' con TimeSeriesSplit(n_splits={cv_splits})..."
        )

        base_pipeline = build_model_pipeline(
            model_name=model_name,
            horizon=horizon,
            include_station_meta=include_station_meta,
        )
        param_dist = PARAM_DISTRIBUTIONS.get(model_name, {})
        eff_n_iter = min(n_iter, _total_combinations(param_dist))

        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_dist,
            n_iter=eff_n_iter,
            scoring="neg_mean_absolute_error",
            cv=tscv,
            random_state=config.RANDOM_SEED,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_tune, y_tune)

        val_pred = search.predict(X_val)
        val_metrics = compute_metrics(y_val, val_pred, hours=val_df["hora"])
        best_params = _strip_model_prefix(search.best_params_)
        cv_trials = _extract_cv_trials(search.cv_results_)

        # Reentrenamiento con train + val (cuidando memoria en random_forest si supera umbral)
        if (
            model_name == "random_forest"
            and max_tuning_rows is not None
            and len(train_val_df) > RANDOM_FOREST_MAX_FIT_ROWS
        ):
            fit_df = train_val_df.iloc[-RANDOM_FOREST_MAX_FIT_ROWS:].reset_index(drop=True)
        else:
            fit_df = train_val_df

        final_pipeline = build_model_pipeline(
            model_name=model_name,
            params=best_params,
            horizon=horizon,
            include_station_meta=include_station_meta,
        )
        final_pipeline.fit(fit_df[feature_cols], fit_df[features.TARGET].to_numpy(np.float64))

        test_pred = final_pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, test_pred, hours=test_df["hora"])
        elapsed = time.perf_counter() - t0

        row = {
            "modelo": model_name,
            "tipo": "ml",
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_wape": val_metrics["wape"],
            "val_r2": val_metrics["r2"],
            "val_mae_horas_pico": val_metrics["mae_horas_pico"],
            "val_wape_horas_pico": val_metrics["wape_horas_pico"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_wape": test_metrics["wape"],
            "test_r2": test_metrics["r2"],
            "test_mae_horas_pico": test_metrics["mae_horas_pico"],
            "test_wape_horas_pico": test_metrics["wape_horas_pico"],
            "tiempo_seg": round(elapsed, 3),
            "mejores_parametros": json.dumps(best_params, ensure_ascii=False),
        }
        results_rows.append(row)

        logger.info(
            f"Modelo '{model_name}' completado en {elapsed:.2f}s | "
            f"val_MAE={val_metrics['mae']:.4f} | test_MAE={test_metrics['mae']:.4f} | "
            f"test_WAPE_pico={test_metrics['wape_horas_pico']:.4f}"
        )

        model_artifacts_data.append(
            {
                "model_name": model_name,
                "best_params": best_params,
                "test_metrics": test_metrics,
                "cv_trials": cv_trials,
                "pipeline": final_pipeline,
            }
        )

        # El campeón se elige con la validación; la prueba solo se reporta, para no usarla en la decisión
        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_test_mae = test_metrics["mae"]
            best_model_name = model_name
            best_pipeline = final_pipeline
            best_test_pred = test_pred
            best_model_params = best_params
            best_model_test_metrics = test_metrics

        del search
        gc.collect()

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = reports_dir / "figures"

    best_pipeline_path = models_dir / BEST_PIPELINE_FILENAME
    joblib.dump(best_pipeline, best_pipeline_path)

    results_df = pd.DataFrame(results_rows).sort_values("test_mae").reset_index(drop=True)
    results_csv_path = reports_dir / RESULTS_CSV_FILENAME
    results_json_path = reports_dir / RESULTS_JSON_FILENAME

    results_df.to_csv(results_csv_path, index=False)
    results_json_path.write_text(
        json.dumps(
            {
                "mejor_modelo": best_model_name,
                "criterio_seleccion": "val_mae",
                "horizonte": horizon,
                "particion": {
                    "filas_train": len(train_df),
                    "filas_val": len(val_df),
                    "filas_test": len(test_df),
                    "val_days": int(val_days),
                    "test_days": int(test_days),
                    "max_train_months": max_train_months,
                },
                "resultados": results_df.to_dict(orient="records"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    comparison_fig_path = plots.model_comparison_plot(results_df, out_dir=figures_dir)
    profile_fig_path = (
        plots.predictions_profile_plot(test_df, best_test_pred, out_dir=figures_dir)
        if best_test_pred is not None
        else None
    )

    champion_registry_info: dict[str, Any] | None = None
    if log_to_mlflow:
        baseline_168_rows = [r for r in results_rows if r["modelo"] == "baseline_estacional_168h"]
        baseline_wape = float(baseline_168_rows[0]["test_wape"]) if baseline_168_rows else None
        shared_artifacts = [
            p
            for p in (results_csv_path, results_json_path, comparison_fig_path, profile_fig_path)
            if p is not None
        ]

        for base_row in [r for r in results_rows if r["tipo"] == "baseline"]:
            tracking.log_model_run(
                model_name=base_row["modelo"],
                metrics={
                    "mae": base_row["test_mae"],
                    "rmse": base_row["test_rmse"],
                    "wape": base_row["test_wape"],
                    "r2": base_row["test_r2"],
                    "mae_horas_pico": base_row["test_mae_horas_pico"],
                    "wape_horas_pico": base_row["test_wape_horas_pico"],
                },
                params=json.loads(base_row["mejores_parametros"]),
                horizon=horizon,
                etiqueta_ventana=etiqueta_ventana,
                max_train_months=max_train_months,
                filas_entrenamiento=len(train_df),
                filas_prueba=len(test_df),
                sha256_tabla=sha256_tabla,
                baseline_wape=baseline_wape,
            )

        best_run_id: str | None = None
        for item in model_artifacts_data:
            run_id = tracking.log_model_run(
                model_name=item["model_name"],
                metrics=item["test_metrics"],
                params=item["best_params"],
                horizon=horizon,
                etiqueta_ventana=etiqueta_ventana,
                max_train_months=max_train_months,
                filas_entrenamiento=len(train_df),
                filas_prueba=len(test_df),
                sha256_tabla=sha256_tabla,
                baseline_wape=baseline_wape,
                pipeline=item["pipeline"],
                X_sample=X_test.head(20),
                cv_trials=item["cv_trials"],
                artifacts=shared_artifacts,
            )
            if item["model_name"] == best_model_name:
                best_run_id = run_id

        if best_run_id is not None and best_model_name is not None:
            champion_registry_info = tracking.register_champion_model(
                run_id=best_run_id,
                model_name=best_model_name,
                metrics=best_model_test_metrics,
                best_params=best_model_params,
                etiqueta_ventana=etiqueta_ventana,
                horizon=horizon,
            )

    logger.success(
        f"Mejor modelo: '{best_model_name}' (test_MAE={best_test_mae:.4f}) guardado en {best_pipeline_path}"
    )

    return {
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
        "best_pipeline_path": best_pipeline_path,
        "results_df": results_df,
        "results_csv_path": results_csv_path,
        "results_json_path": results_json_path,
        "comparison_fig_path": comparison_fig_path,
        "profile_fig_path": profile_fig_path,
        "champion_registry_info": champion_registry_info,
        "split_sizes": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
    }


def _extract_cv_trials(cv_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Convierte `cv_results_` de `RandomizedSearchCV` en una lista de ensayos para MLflow."""
    trials: list[dict[str, Any]] = []
    params_list = cv_results.get("params", [])
    mean_scores = cv_results.get("mean_test_score", [])
    std_scores = cv_results.get("std_test_score", [])
    for idx, raw_params in enumerate(params_list):
        clean_params = _strip_model_prefix(raw_params)
        mean_mae = float(-mean_scores[idx]) if idx < len(mean_scores) else 0.0
        std_mae = float(std_scores[idx]) if idx < len(std_scores) else 0.0
        trials.append(
            {
                "params": clean_params,
                "metrics": {"cv_mae": mean_mae, "cv_mae_std": std_mae},
            }
        )
    return trials


@app.command()
def main(
    dataset_path: Path = config.PROCESSED_DATA_DIR / DEFAULT_DATASET_FILENAME,
    models_dir: Path = config.MODELS_DIR,
    reports_dir: Path = config.REPORTS_DIR,
    horizon: str = config.DEFAULT_HORIZON,
    max_train_months: int = DEFAULT_MAX_TRAIN_MONTHS,
    val_days: int = config.VAL_DAYS,
    test_days: int = config.TEST_DAYS,
    n_iter: int = DEFAULT_N_ITER,
    cv_splits: int = DEFAULT_CV_SPLITS,
    max_tuning_rows: int = DEFAULT_MAX_TUNING_ROWS,
    include_station_meta: bool = True,
    log_to_mlflow: bool = False,
) -> None:
    """Ejecuta desde CLI el flujo completo de entrenamiento, ajuste y evaluación de modelos."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"No se encontró la tabla procesada en {dataset_path}. "
            "Ejecute primero `python -m mibici_demand.cli construir`."
        )

    logger.info(f"Cargando tabla de entrenamiento desde {dataset_path}...")
    table = pd.read_parquet(dataset_path)
    tune_and_evaluate(
        table=table,
        horizon=horizon,
        include_station_meta=include_station_meta,
        val_days=val_days,
        test_days=test_days,
        max_train_months=max_train_months,
        n_iter=n_iter,
        cv_splits=cv_splits,
        max_tuning_rows=max_tuning_rows,
        models_dir=models_dir,
        reports_dir=reports_dir,
        log_to_mlflow=log_to_mlflow,
    )


if __name__ == "__main__":
    app()

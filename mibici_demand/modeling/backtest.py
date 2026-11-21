"""Backtesting diario a 24 h con tres reentrenamientos y registro separado en MLflow.

Ejecutar: python -m mibici_demand.modeling.backtest
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mibici_demand import config, dataset, features
from mibici_demand.modeling.pipeline import build_model_pipeline
from mibici_demand.modeling.train import compute_metrics
from mibici_demand.temporal import BacktestSpec, build_fold, prepare_data

EXPERIMENT = "mibici-fase3-validacion-temporal"
MODEL = "hist_gbm_poisson"
BASELINES = {"baseline_diario_24h": "lag_24", "baseline_semanal_168h": "lag_168"}
# Se fijan antes de ejecutar los cortes; no se recuperan pesos ni parámetros del campeón.
MODEL_PARAMS = {
    "max_iter": 100,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "learning_rate": 0.1,
    "l2_regularization": 0.0,
    "early_stopping": False,
}


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def format_metric(value: object) -> str:
    return "NA" if pd.isna(value) else f"{value:.4f}"


def evaluation_metrics(y: pd.Series, prediction: np.ndarray, hours: pd.Series) -> dict:
    """Reutiliza las métricas históricas y marca denominadores/varianzas nulos como NA."""
    metrics = compute_metrics(y, prediction, hours=hours)
    if float(y.sum()) == 0:
        metrics["wape"] = None
    if len(y) < 2 or y.nunique() < 2:
        metrics["r2"] = None
    peak = y.loc[hours.isin(config.PEAK_HOURS)]
    if float(peak.sum()) == 0:
        metrics["wape_horas_pico"] = None
    return metrics


def fit_fold(train: pd.DataFrame, *, threads: int = 4, params: dict | None = None):
    """Instancia y ajusta un objeto nuevo, sin cargar el pipeline histórico."""
    pipeline = build_model_pipeline(
        MODEL, params={**MODEL_PARAMS, **(params or {})}, horizon="24h", include_station_meta=False
    )
    columns = features.get_feature_columns(train, horizon="24h")
    with threadpool_limits(limits=threads):
        pipeline.fit(train[columns], train["demanda"])
    return pipeline, columns


def _environment() -> dict:
    sources = [
        *sorted((config.ROOT / "mibici_demand").rglob("*.py")),
        config.ROOT / "pyproject.toml",
        config.ROOT / "requirements.txt",
    ]
    return {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "paquetes": {
            d.metadata["Name"]: d.version
            for d in sorted(
                importlib.metadata.distributions(), key=lambda d: d.metadata["Name"].lower()
            )
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True
        ).strip(),
        "git_branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=config.ROOT, text=True
        ).strip(),
        "git_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=config.ROOT, text=True
            ).strip()
        ),
        "sha256_codigo": {
            p.relative_to(config.ROOT).as_posix(): dataset.sha256_file(p) for p in sources
        },
    }


def _log_fold(
    pipeline, train, evaluation, columns, audit, rows, provenance, environment, processed_dir
):
    mlflow.log_params(
        {
            **{k: v for k, v in audit.items() if not isinstance(v, (list, dict))},
            **pipeline.named_steps["modelo"].get_params(),
            "modelo": MODEL,
            "horizonte": "24h",
            "emision": "diaria_00:00",
            "random_seed": config.RANDOM_SEED,
            "python": environment["python"],
            "scikit_learn": environment["paquetes"]["scikit-learn"],
            "catalogo_actual": False,
        }
    )
    mlflow.set_tags(
        {
            "stage": "fase-3",
            "git_commit": environment["git_commit"],
            "git_branch": environment["git_branch"],
            "tipo": "corte_temporal",
        }
    )
    for row in rows:
        mlflow.log_metrics(
            {
                f"{row['modelo']}.{k}": float(row[k])
                for k in ("mae", "rmse", "wape", "r2", "mae_horas_pico", "wape_horas_pico")
                if row[k] is not None
            }
        )
    for frame, context in ((train, "training"), (evaluation, "evaluation")):
        data = mlflow.data.from_pandas(
            frame[columns + ["demanda"]],
            source=str(processed_dir / "viajes_limpios.parquet"),
            name=f"{context}_{audit['corte'][:10]}",
        )
        mlflow.log_input(data, context=context)
    mlflow.log_dict(audit, "auditoria_corte.json")
    mlflow.log_dict(provenance, "procedencia_datos.json")
    mlflow.log_dict(environment, "entorno.json")
    requirements = [
        f"{p}=={environment['paquetes'][p]}"
        for p in ("scikit-learn", "numpy", "scipy", "pandas", "joblib", "cloudpickle", "mlflow")
    ]
    sample = train[columns].head(5)
    logged = mlflow.sklearn.log_model(
        sk_model=pipeline,
        name="pipeline",
        input_example=sample,
        signature=infer_signature(sample, pipeline.predict(sample)),
        code_paths=[str(config.ROOT / "mibici_demand")],
        pip_requirements=requirements,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    return logged.model_uri


def make_plots(metrics: pd.DataFrame, daily: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, ("mae", "rmse", "wape", "r2")):
        for model, rows in metrics.groupby("modelo", sort=False):
            axis.plot(rows["corte"].str[:10], rows[metric], marker="o", label=model)
        axis.set_title(metric.upper() if metric != "r2" else "R²")
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=15)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("MiBici: tres cortes temporales, pronóstico diario a 24 h")
    fig.savefig(output_dir / "metricas_por_corte.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for model, rows in daily.groupby("modelo", sort=False):
        dates = pd.to_datetime(rows["dia"])
        axes[0].plot(dates, rows["wape"], marker=".", label=model)
        axes[1].plot(dates, rows["prediccion_total"], label=model)
    observed = daily.drop_duplicates("dia").sort_values("dia")
    axes[1].plot(
        pd.to_datetime(observed["dia"]),
        observed["demanda_total"],
        color="black",
        linewidth=2,
        label="observado",
    )
    axes[0].set_ylabel("WAPE diario")
    axes[1].set_ylabel("Viajes en la rejilla evaluada")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        axis.tick_params(axis="x", rotation=20)
    fig.savefig(output_dir / "evolucion_diaria.png", dpi=150)
    plt.close(fig)


def write_report(
    output_dir: Path,
    metrics: pd.DataFrame,
    pooled: pd.DataFrame,
    audits: list[dict],
    execution: dict,
) -> None:
    lines = [
        "# Fase 3: preparación reproducible y validación temporal",
        "",
        f"Estado: **EJECUTADO**. Ejecución `{execution['id']}`. Python {execution['entorno']['python']}; rama `{execution['entorno']['git_branch']}`.",
        "",
        "## Datos y reproducción",
        "",
        "Se descargaron/verificaron solo los meses necesarios (agosto y septiembre de 2026 en la configuración predeterminada), desde [MiBici](https://mibici.net/en/open-data/). Cada CSV debe coincidir en tamaño y SHA-256 con `data/raw/manifest.csv` de Fase 1. El manifiesto original no se reescribe. `procedencia_datos.json` contiene URLs, hashes y ventana utilizada; `limpieza.csv`, los conteos por regla. Los datos personales no se conservan en el parquet preparado.",
        "",
        "```powershell",
        r".venv\Scripts\python.exe -m mibici_demand.modeling.backtest",
        r"# Repetir sin descargar: añadir --sin-descarga",
        r".venv\Scripts\python.exe -m pytest tests -q",
        r".venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow_fase3.db",
        "```",
        "",
        "Los comandos se ejecutan desde la raíz en la rama `fase3`. Se reutilizan `dataset.download/read_trips`, `TripCleaner`, `features.monthly_tables/get_feature_columns`, `build_model_pipeline` y `compute_metrics`. Datos en `data/raw/fase3` y `data/processed/fase3`; modelos nuevos en `models/fase3/<ejecución>`; MLflow en una base y experimento separados. `versiones.json` fija el entorno y hashes del código (incluye archivos todavía sin commit). No se modifica el campeón ni los resultados de Fases 1/2.",
        "",
        "## Metodología y protección temporal",
        "",
        "Cada corte entrena un pipeline independiente. El estimador es HistGradientBoostingRegressor con pérdida Poisson, parámetros fijados **antes** de observar estas ventanas, sin búsqueda ni selección del mejor corte. No se cargan pesos ni hiperparámetros optimizados del modelo histórico. Parámetros, semilla, fechas, fuentes y modelos serializados se registran por corte en MLflow.",
        "",
        "El entrenamiento abarca 28 días en la configuración predeterminada, con una purga/maduración adicional de 24 horas antes del corte. Así, incluso un viaje válido de la duración máxima permitida (24 h) termina antes de usar su etiqueta. Todas las divisiones son intervalos semiabiertos por timestamps locales completos, nunca por posiciones de filas: `[inicio, fin)`. Las estaciones de una misma hora no se reparten entre entrenamiento y evaluación.",
        "",
        "En cada semana se emite un pronóstico a las 00:00 de cada día para sus 24 horas siguientes. El modelo se mantiene fijo durante esa semana; los rezagos se actualizan con observaciones anteriores a cada nueva emisión. Esto es backtesting de pronósticos **diarios**, no un pronóstico único de siete días. Cada `lag_24`/`lag_168` cuenta exclusivamente viajes terminados antes de su origen de pronóstico y de una hora ya cerrada. Los viajes todavía abiertos no se cuentan. No se usan `lag_1` ni `media_24h`. Las etiquetas finales de evaluación solo se usan para puntuar.",
        "",
        "El perfil por estación, PCA, imputador, escalador y codificadores se ajustan solo con el entrenamiento del corte. La cohorte se fija con estaciones conocidas, con al menos siete días de historial y actividad durante la semana anterior al corte. Las horas de servicio se aprenden con el entrenamiento (umbral heredado 0.001 de viajes). La cohorte y el horario permanecen fijos al evaluar; no se consultan últimas apariciones ni aperturas futuras. El catálogo actual de capacidad/coordenadas se excluye por carecer de versiones históricas verificables.",
        "",
        "## Ventanas ejecutadas",
        "",
        "| Corte / inicio evaluación | Inicio train | Fin train exclusivo | Fin evaluación exclusivo | Filas train | Filas evaluación | Estaciones | Cobertura de viajes |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for a in audits:
        lines.append(
            f"| {a['corte'][:10]} | {a['train_inicio'][:10]} | {a['train_fin_exclusivo'][:10]} | {a['evaluacion_fin_exclusivo'][:10]} | {a['filas_train']:,} | {a['filas_evaluacion']:,} | {a['estaciones']} | {a['cobertura_demanda']:.2%} |"
        )
    lines += [
        "",
        "`cortes.json` conserva timestamps completos, IDs de estaciones, horario y exclusiones por corte. La cobertura es la proporción de viajes reales de la red incluidos en la rejilla, no una medida de precisión.",
        "",
        "## Resultados reales",
        "",
        "WAPE se expresa como proporción (0.50 equivale a 50%). R² y WAPE se dejan sin valor si no tienen varianza/denominador válidos. Modelo y baselines comparten exactamente las mismas filas.",
        "",
        "| Corte | Modelo | MAE | RMSE | WAPE | R² | WAPE pico |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics.to_dict("records"):
        values = [
            "NA" if pd.isna(row[k]) else f"{row[k]:.4f}"
            for k in ("mae", "rmse", "wape", "r2", "wape_horas_pico")
        ]
        lines.append(f"| {row['corte'][:10]} | {row['modelo']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "### Comparación conjunta de las tres semanas",
        "",
        "Las métricas conjuntas se recalculan concatenando todas las predicciones; no son un promedio simple de WAPE o R².",
        "",
        "| Modelo | MAE | RMSE | WAPE | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in pooled.to_dict("records"):
        lines.append(
            f"| {row['modelo']} | "
            + " | ".join(format_metric(row[k]) for k in ("mae", "rmse", "wape", "r2"))
            + " |"
        )
    lines += ["", "### Cambio al avanzar el corte", ""]
    for cutoff, group in metrics.groupby("corte", sort=False):
        model_wape = group.loc[group["modelo"] == MODEL, "wape"].iloc[0]
        best_baseline = group.loc[group["modelo"] != MODEL].sort_values("wape").iloc[0]
        improvement = (
            100 * (1 - model_wape / best_baseline["wape"])
            if pd.notna(model_wape)
            and pd.notna(best_baseline["wape"])
            and best_baseline["wape"] > 0
            else None
        )
        lines.append(
            f"- {cutoff[:10]}: WAPE del modelo {format_metric(model_wape)}; mejor baseline `{best_baseline['modelo']}` ({format_metric(best_baseline['wape'])}); mejora relativa {format_metric(improvement)}% (negativa significa empeoramiento)."
        )
    lines += [
        "",
        "### Errores grandes y lectura del resultado",
        "",
        "Mejorar MAE/WAPE no garantiza mejorar RMSE/R². Se conservan todas las predicciones, sin descartar casos difíciles ni limitar sus valores después de observar los resultados. `errores_mayores.csv` muestra los veinte errores absolutos más grandes de cada corte.",
        "",
    ]
    for cutoff, group in metrics.groupby("corte", sort=False):
        m = group.loc[group["modelo"] == MODEL].iloc[0]
        weekly = group.loc[group["modelo"] == "baseline_semanal_168h"].iloc[0]
        if m["rmse"] > weekly["rmse"]:
            lines.append(
                f"- En {cutoff[:10]}, el RMSE del modelo ({m['rmse']:.4f}) supera al semanal ({weekly['rmse']:.4f}); R² del modelo: {format_metric(m['r2'])}. Este resultado desfavorable impide afirmar que el modelo domina todas las métricas."
            )
    extremes = pd.read_csv(output_dir / "errores_mayores.csv")
    largest = extremes.sort_values("error_absoluto", ascending=False).iloc[0]
    lines.append(
        f"- Mayor error observado: estación {int(largest['estacion'])}, hora {largest['hora_inicio']}; observado {int(largest['demanda'])}, predicción {largest[MODEL]:.4f}, lag diario {largest['lag_24']:.0f} y semanal {largest['lag_168']:.0f}. Es un diagnóstico descriptivo; no se atribuye una causa única ni se ajustan hiperparámetros usando estas semanas."
    )
    lines += [
        "",
        "![Métricas por corte](metricas_por_corte.png)",
        "",
        "![Evolución diaria](evolucion_diaria.png)",
        "",
        "## Trazabilidad y observaciones del profesor",
        "",
        f"Experimento MLflow: `{EXPERIMENT}`. Corrida padre: `{execution['mlflow_parent_run_id']}`. `ejecucion.json` enlaza los tres runs hijos y los modelos; se registran MAE, RMSE, WAPE, R² y métricas pico de los tres predictores, firmas de entrada y datasets de entrenamiento/evaluación.",
        "",
        "Se atendió la solicitud de **avanzar los cortes y comparar resultados** con tres reentrenamientos cronológicos y tres semanas posteriores distintas. Se añadió la comparación diaria/semanal sobre las mismas observaciones y controles automáticos de timestamps, maduración de etiquetas, disponibilidad de rezagos y preprocesamiento. La evidencia es ejecutada sobre CSV reales, sin trasladar las cifras históricas de Fase 2 a estas ventanas.",
        "",
        "## Limitaciones y pendientes",
        "",
        "- Los CSV se publican mensualmente. La disponibilidad diaria de viajes terminados se **simula offline**; esta fuente pública por sí sola no permite operar un servicio diario en tiempo real. Hace falta un feed operativo con disponibilidad equivalente para desplegarlo.",
        "- Solo se estudian tres semanas de septiembre de 2026 y una ventana corta de entrenamiento por memoria/alcance. No demuestra estabilidad anual, causalidad de los cambios ni significancia estadística; no se calculan intervalos de confianza.",
        "- La familia de modelo se eligió por la experiencia de Fase 2. Aunque cada ajuste usa solo el pasado y aquí no se optimizan hiperparámetros con las ventanas evaluadas, estas semanas no constituyen un holdout totalmente ajeno al trabajo histórico. Una evaluación prospectiva posterior requiere datos no usados en la selección previa.",
        "- Estaciones nuevas o sin suficiente historial se excluyen de la cohorte congelada; la cobertura y los IDs excluidos están documentados. No hay un calendario externo verificado de aperturas/cierres; un cero posterior puede reflejar falta de demanda o una interrupción de servicio.",
        "- Los conteos de rezagos pueden ser provisionales por viajes aún abiertos; las etiquetas evaluadas están finalizadas. No se supone que sus valores sean idénticos a los rezagos de la tabla histórica.",
        "- Las cifras no se comparan directamente con Fase 2: cambian ventanas, datos, cohorte, disponibilidad de variables e hiperparámetros. No se promociona ningún nuevo campeón.",
        "- FastAPI, Docker y drift quedan fuera de este bloque. Tampoco se ha verificado todavía ejecución Linux.",
        "",
    ]
    (output_dir / "validacion_temporal.md").write_text("\n".join(lines), encoding="utf-8")


def run_backtest(
    trips: pd.DataFrame,
    spec: BacktestSpec,
    provenance: dict,
    output_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    *,
    threads: int = 4,
    log_to_mlflow: bool = True,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    execution_id = (
        datetime.now(ZoneInfo("America/Mexico_City")).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid4().hex[:8]
    )
    model_dir = models_dir / execution_id
    model_dir.mkdir(parents=True, exist_ok=True)
    execution = {
        "id": execution_id,
        "estado": "en_ejecucion",
        "configuracion": asdict(spec),
        "parametros_modelo": MODEL_PARAMS,
        "threads": threads,
        "entorno": environment,
        "mlflow_parent_run_id": None,
        "cortes": [],
    }
    write_json(output_dir / "estado_actual.json", {"estado": "en_ejecucion", "id": execution_id})
    write_json(output_dir / "versiones.json", environment)
    if log_to_mlflow:
        uri = f"sqlite:///{(config.ROOT / 'mlflow_fase3.db').as_posix()}"
        mlflow.set_tracking_uri(uri)
        client = MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name(EXPERIMENT)
        experiment_id = (
            experiment.experiment_id
            if experiment
            else client.create_experiment(
                EXPERIMENT, artifact_location=(config.ROOT / "mlartifacts/fase3").as_uri()
            )
        )
        parent = mlflow.start_run(experiment_id=experiment_id, run_name=f"backtest-{execution_id}")
    else:
        parent = nullcontext(None)
    metrics_rows, daily_rows, audits, predictions = [], [], [], []
    with parent as parent_run:
        if parent_run:
            execution["mlflow_parent_run_id"] = parent_run.info.run_id
            mlflow.set_tags(
                {
                    "stage": "fase-3",
                    "git_branch": environment["git_branch"],
                    "git_commit": environment["git_commit"],
                    "tipo": "backtesting_tres_cortes",
                }
            )
            mlflow.log_params(
                {
                    **asdict(spec),
                    **MODEL_PARAMS,
                    "random_seed": config.RANDOM_SEED,
                    "threads": threads,
                }
            )
        for cutoff in spec.timestamps:
            print(f"Construyendo corte {cutoff}", flush=True)
            train, evaluation, audit = build_fold(trips, cutoff, spec)
            total = int(
                (
                    (trips["Inicio_del_viaje"] >= cutoff)
                    & (trips["Inicio_del_viaje"] < cutoff + spec.evaluation_delta)
                ).sum()
            )
            audit.update(
                {
                    "demanda_total_red_evaluacion": total,
                    "demanda_en_rejilla_evaluada": int(evaluation["demanda"].sum()),
                    "cobertura_demanda": float(evaluation["demanda"].sum() / total)
                    if total
                    else 0.0,
                }
            )
            child = (
                mlflow.start_run(
                    experiment_id=experiment_id, run_name=f"corte-{cutoff.date()}", nested=True
                )
                if log_to_mlflow
                else nullcontext(None)
            )
            with child as child_run:
                print(
                    f"Entrenando pipeline nuevo: {len(train):,} filas; evaluando {len(evaluation):,}",
                    flush=True,
                )
                pipeline, columns = fit_fold(train, threads=threads)
                with threadpool_limits(limits=threads):
                    prediction = np.clip(pipeline.predict(evaluation[columns]), 0.0, None)
                if not np.isfinite(prediction).all():
                    raise ValueError("Predicciones no finitas; la evaluación no es válida.")
                evaluated = evaluation.copy()
                evaluated[MODEL] = prediction
                evaluated["corte"] = str(cutoff)
                for baseline, lag in BASELINES.items():
                    evaluated[baseline] = evaluated[lag]
                rows = []
                for model in (MODEL, *BASELINES):
                    row = {
                        "corte": str(cutoff),
                        "modelo": model,
                        "filas": len(evaluated),
                        **evaluation_metrics(
                            evaluated["demanda"], evaluated[model].to_numpy(), evaluated["hora"]
                        ),
                        "mlflow_run_id": child_run.info.run_id if child_run else None,
                    }
                    rows.append(row)
                    for day, day_frame in evaluated.groupby("origen_pronostico", sort=True):
                        daily_rows.append(
                            {
                                "corte": str(cutoff),
                                "dia": str(day),
                                "modelo": model,
                                "filas": len(day_frame),
                                "demanda_total": int(day_frame["demanda"].sum()),
                                "prediccion_total": float(day_frame[model].sum()),
                                **evaluation_metrics(
                                    day_frame["demanda"],
                                    day_frame[model].to_numpy(),
                                    day_frame["hora"],
                                ),
                            }
                        )
                artifact = model_dir / f"pipeline_{cutoff:%Y%m%d}.joblib"
                joblib.dump(pipeline, artifact)
                audit["modelo_sha256"] = dataset.sha256_file(artifact)
                audit["columnas_entrada"] = columns
                model_uri = (
                    _log_fold(
                        pipeline,
                        train,
                        evaluation,
                        columns,
                        audit,
                        rows,
                        provenance,
                        environment,
                        processed_dir,
                    )
                    if log_to_mlflow
                    else None
                )
                execution["cortes"].append(
                    {
                        "corte": str(cutoff),
                        "run_id": child_run.info.run_id if child_run else None,
                        "modelo_local": artifact.relative_to(config.ROOT).as_posix()
                        if artifact.is_relative_to(config.ROOT)
                        else str(artifact),
                        "mlflow_model_uri": model_uri,
                    }
                )
                metrics_rows.extend(rows)
                audits.append(audit)
                predictions.append(evaluated)
                print(json.dumps(rows, ensure_ascii=False), flush=True)
            del pipeline, train
        metrics = pd.DataFrame(metrics_rows)
        daily = pd.DataFrame(daily_rows)
        all_predictions = pd.concat(predictions, ignore_index=True)
        pooled = pd.DataFrame(
            [
                {
                    "modelo": m,
                    "filas": len(all_predictions),
                    **evaluation_metrics(
                        all_predictions["demanda"],
                        all_predictions[m].to_numpy(),
                        all_predictions["hora"],
                    ),
                }
                for m in (MODEL, *BASELINES)
            ]
        )
        metrics.to_csv(output_dir / "metricas_cortes.csv", index=False)
        daily.to_csv(output_dir / "metricas_diarias.csv", index=False)
        pooled.to_csv(output_dir / "metricas_conjuntas.csv", index=False)
        write_json(output_dir / "metricas_cortes.json", metrics_rows)
        write_json(output_dir / "cortes.json", audits)
        errors = all_predictions.copy()
        errors["error_absoluto"] = (errors["demanda"] - errors[MODEL]).abs()
        errors = (
            errors.sort_values("error_absoluto", ascending=False)
            .groupby("corte", sort=False)
            .head(20)
        )
        errors[
            [
                "corte",
                "origen_pronostico",
                "estacion",
                "hora_inicio",
                "demanda",
                "lag_24",
                "lag_168",
                MODEL,
                "error_absoluto",
            ]
        ].to_csv(output_dir / "errores_mayores.csv", index=False)
        processed_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = processed_dir / "predicciones_backtest.parquet"
        all_predictions.to_parquet(predictions_path, index=False)
        execution.update(
            {
                "estado": "completado",
                "filas_predicciones": len(all_predictions),
                "predicciones_sha256": dataset.sha256_file(predictions_path),
                "procedencia": provenance,
            }
        )
        make_plots(metrics, daily, output_dir)
        write_report(output_dir, metrics, pooled, audits, execution)
        write_json(output_dir / "ejecucion.json", execution)
        write_json(output_dir / "estado_actual.json", {"estado": "completado", "id": execution_id})
        if parent_run:
            for row in pooled.to_dict("records"):
                mlflow.log_metrics(
                    {
                        f"conjunto.{row['modelo']}.{k}": float(row[k])
                        for k in ("mae", "rmse", "wape", "r2")
                        if row[k] is not None
                    }
                )
            mlflow.log_artifacts(str(output_dir), artifact_path="reportes")
            for source in (config.ROOT / "mibici_demand").rglob("*.py"):
                mlflow.log_artifact(
                    str(source),
                    artifact_path="codigo/" + source.parent.relative_to(config.ROOT).as_posix(),
                )
    return execution


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cortes", nargs=3, default=list(BacktestSpec().cutoffs))
    parser.add_argument("--train-days", type=int, default=28)
    parser.add_argument("--evaluation-days", type=int, default=7)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sin-descarga", action="store_true")
    args = parser.parse_args(argv)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=config.ROOT, text=True
    ).strip()
    if branch != "fase3":
        raise SystemExit(f"Este bloque solo se ejecuta en la rama fase3; rama actual: {branch}")
    execution_folder = datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    output = config.REPORTS_DIR / "fase3/ejecuciones" / execution_folder
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.threads < 1:
            raise ValueError("threads debe ser positivo")
        spec = BacktestSpec(tuple(args.cortes), args.train_days, args.evaluation_days)
        processed = config.PROCESSED_DIR / "fase3/ejecuciones" / execution_folder
        trips, provenance = prepare_data(
            spec,
            config.RAW_DIR / "fase3",
            processed,
            output,
            download_missing=not args.sin_descarga,
        )
        execution = run_backtest(
            trips,
            spec,
            provenance,
            output,
            processed,
            config.MODELS_DIR / "fase3",
            threads=args.threads,
        )
        print(f"Completado: {execution['id']} -> {output}")
    except Exception as error:
        write_json(output / "estado_actual.json", {"estado": "bloqueado", "error": str(error)})
        write_json(
            output / "bloqueo.json",
            {
                "estado": "bloqueado",
                "error": type(error).__name__,
                "detalle": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        (output / "validacion_temporal.md").write_text(
            f"# Validación temporal bloqueada\n\nNo se declara una validación exitosa.\n\n`{type(error).__name__}: {error}`\n\nConsultar `bloqueo.json` y los logs de ejecución.\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()

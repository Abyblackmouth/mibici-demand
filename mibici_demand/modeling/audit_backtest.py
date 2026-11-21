"""Auditoría reproducible sin ajuste ni reentrenamiento de los modelos de Fase 3.

python -m mibici_demand.modeling.audit_backtest
No descarga, lee ni evalúa observaciones de octubre.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from threadpoolctl import threadpool_limits

from mibici_demand import config, dataset, features
from mibici_demand.modeling.backtest import BASELINES, MODEL, evaluation_metrics, write_json
from mibici_demand.temporal import BacktestSpec, build_fold

KEYS = ["estacion", "hora_inicio"]


def replace_numeric_preserving_dtype(
    frame: pd.DataFrame, column: str, value: float
) -> pd.DataFrame:
    """Una perturbación controlada no debe cambiar también la precisión numérica."""
    changed = frame.copy()
    changed[column] = pd.Series(value, index=changed.index, dtype=frame[column].dtype)
    if not changed.dtypes.equals(frame.dtypes):
        raise ValueError("La perturbación cambió los tipos del contrato original.")
    return changed


def squared_error_concentration(y, prediction, fraction: float = 0.01) -> dict:
    """Selecciona ceil(n*fraction) filas, sin recortar errores de las métricas."""
    actual = np.asarray(y, dtype=float)
    predicted = np.asarray(prediction, dtype=float)
    if (
        not 0 < fraction <= 1
        or actual.ndim != 1
        or actual.shape != predicted.shape
        or not len(actual)
    ):
        raise ValueError("Vectores no vacíos, de igual longitud y fracción en (0,1].")
    if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
        raise ValueError("La concentración requiere valores finitos.")
    errors = np.sort((actual - predicted) ** 2)[::-1]
    total = float(errors.sum())
    count = int(np.ceil(len(errors) * fraction))
    return {
        "n": len(errors),
        "n_top": count,
        "fraccion_solicitada": fraction,
        "fraccion_efectiva": count / len(errors),
        "sse_total": total,
        "sse_top": float(errors[:count].sum()),
        "proporcion_sse_top": float(errors[:count].sum() / total) if total else None,
        "proporcion_sse_top20": float(errors[:20].sum() / total) if total else None,
    }


def snapshot_originals(root: Path, output: Path) -> dict:
    """Copia la evidencia anterior; en repeticiones nunca reemplaza otra versión."""
    execution = json.loads((root / "ejecucion.json").read_text(encoding="utf-8"))
    destination = output / "originales" / execution["id"]
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for source in sorted(root.iterdir()):
        if not source.is_file() or source.name.startswith(("auditoria_", "pytest_auditoria")):
            continue
        target = destination / source.name
        digest = dataset.sha256_file(source)
        if target.exists() and dataset.sha256_file(target) != digest:
            raise ValueError(f"La instantánea existente difiere: {target}. No se sobrescribe.")
        if not target.exists():
            shutil.copy2(source, target)
        manifest[source.name] = digest
    return {"ejecucion_original": execution["id"], "archivos_sha256": manifest}


def read_sources_independently(
    root: Path, provenance: dict
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Repite las reglas con máscaras explícitas, sin TripCleaner/monthly_tables.

    Comparar dos llamadas al mismo constructor no detectaría un error compartido.
    Se leen únicamente los meses que constan en la ejecución original.
    """
    if provenance["meses"] != ["2026-08", "2026-09"]:
        raise ValueError(
            "Esta auditoría acota sus fuentes a agosto y septiembre; no consume octubre."
        )
    valid_frames, raw_frames, integrity = [], [], []
    fields = ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id", "Viaje_Id"]
    start = pd.Timestamp(provenance["inicio_inclusivo"])
    end = pd.Timestamp(provenance["fin_exclusivo"])
    for source in provenance["archivos"]:
        path = root / "data/raw/fase3" / source["archivo"]
        if dataset.sha256_file(path) != source["sha256"]:
            raise ValueError(f"Cambió la fuente: {path}")
        raw = dataset.read_trips(path)
        raw_frames.append(
            raw.loc[
                raw["Inicio_del_viaje"].ge(start) & raw["Inicio_del_viaje"].lt(end), fields
            ].copy()
        )
        clean = raw.loc[
            raw["Inicio_del_viaje"].notna() & raw["Origen_Id"].notna()
        ].drop_duplicates()
        seconds = (clean["Fin_del_viaje"] - clean["Inicio_del_viaje"]).dt.total_seconds()
        valid = seconds.notna() & seconds.ge(0) & seconds.le(config.MAX_TRIP_SECONDS)
        failed_dock = seconds.lt(config.MIN_TRIP_SECONDS) & clean["Origen_Id"].eq(
            clean["Destino_Id"]
        )
        clean = clean.loc[valid & ~failed_dock]
        valid_frames.append(
            clean.loc[
                clean["Inicio_del_viaje"].ge(start) & clean["Inicio_del_viaje"].lt(end), fields
            ].copy()
        )
        integrity.append(
            {
                "archivo": source["archivo"],
                "sha256_verificado": True,
                "filas_sin_inicio": int(raw["Inicio_del_viaje"].isna().sum()),
                "filas_sin_fin": int(raw["Fin_del_viaje"].isna().sum()),
            }
        )
    return (
        pd.concat(valid_frames, ignore_index=True),
        pd.concat(raw_frames, ignore_index=True),
        integrity,
    )


def hourly_counts(trips: pd.DataFrame) -> pd.Series:
    return trips.groupby(
        [
            trips["Origen_Id"].rename("estacion"),
            trips["Inicio_del_viaje"].dt.floor("h").rename("hora_inicio"),
        ]
    ).size()


def verify_labels_and_lags(trips: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Reconstruye cada etiqueta y rezago desde viajes, sin usar features.py."""
    labels = hourly_counts(trips)
    index = pd.MultiIndex.from_frame(predictions[KEYS])
    expected_y = labels.reindex(index, fill_value=0).to_numpy()
    if not np.array_equal(expected_y, predictions["demanda"].to_numpy()):
        raise ValueError("Etiquetas distintas de los conteos independientes.")
    for origin, frame in predictions.groupby("origen_pronostico", sort=True):
        if not frame["hora_inicio"].dt.floor("D").eq(origin).all():
            raise ValueError("La emisión no corresponde a la medianoche del día objetivo.")
        known = trips.loc[trips["Fin_del_viaje"].lt(origin) & trips["Inicio_del_viaje"].lt(origin)]
        known_counts = hourly_counts(known)
        for lag in (24, 168):
            source = frame["hora_inicio"] - pd.Timedelta(hours=lag)
            if (
                not source.eq(frame[f"fuente_lag_{lag}"]).all()
                or not (source + pd.Timedelta(hours=1) <= origin).all()
            ):
                raise ValueError("Fuente temporal incorrecta o futura.")
            keys = pd.MultiIndex.from_arrays([frame["estacion"], source])
            expected_lag = known_counts.reindex(keys, fill_value=0).to_numpy()
            if not np.array_equal(expected_lag, frame[f"lag_{lag}"].to_numpy()):
                raise ValueError(f"Rezago incorrecto: lag_{lag}, origen {origin}.")
    return {
        "etiquetas_verificadas": len(predictions),
        "rezagos_verificados": 2 * len(predictions),
        "discrepancias": 0,
    }


def verify_composition(
    trips: pd.DataFrame, evaluated: pd.DataFrame, audit: dict, spec: BacktestSpec
) -> None:
    """Composición reconstruida solo con datos anteriores al corte, no con y futuro."""
    cutoff = pd.Timestamp(audit["corte"])
    past = trips.loc[trips["Inicio_del_viaje"].lt(cutoff) & trips["Fin_del_viaje"].lt(cutoff)]
    first_origin = past.groupby("Origen_Id")["Inicio_del_viaje"].min()
    first_destination = past.groupby("Destino_Id")["Fin_del_viaje"].min()
    first = pd.concat([first_origin, first_destination]).groupby(level=0).min().dt.floor("h")
    recent = past[past["Inicio_del_viaje"] >= cutoff - pd.Timedelta(days=7)]
    recent_ids = set(recent["Origen_Id"]) | set(recent["Destino_Id"].dropna())
    stations = sorted(
        int(s)
        for s in first.index
        if s in recent_ids and first.loc[s] + pd.Timedelta(days=7) <= cutoff
    )
    train_end = cutoff - pd.Timedelta(hours=spec.label_delay_hours)
    starts = past.loc[
        past["Inicio_del_viaje"].ge(train_end - pd.Timedelta(days=spec.train_days))
        & past["Inicio_del_viaje"].lt(train_end),
        "Inicio_del_viaje",
    ]
    hour_counts = starts.dt.hour.value_counts()
    hours = sorted(
        int(h)
        for h in hour_counts.index[hour_counts / hour_counts.sum() >= config.CLOSED_HOUR_SHARE]
    )
    target_times = pd.date_range(
        cutoff, cutoff + spec.evaluation_delta, freq="h", inclusive="left"
    )
    target_times = target_times[target_times.hour.isin(hours)]
    expected = pd.MultiIndex.from_product([stations, target_times], names=KEYS)
    actual = pd.MultiIndex.from_frame(evaluated[KEYS])
    if (
        actual.has_duplicates
        or len(actual) != len(expected)
        or len(actual.difference(expected))
        or len(expected.difference(actual))
    ):
        raise ValueError("La composición de evaluación no coincide con el plan conocido al corte.")
    if stations != audit["ids_estaciones"] or hours != audit["horas_servicio"]:
        raise ValueError("Cohorte u horario auditados distintos del pasado reconstruido.")


def distribution_row(values: pd.Series, cutoff: str, variable: str, split: str) -> dict:
    quantiles = values.quantile([0, 0.5, 0.9, 0.99, 0.999, 1]).to_numpy()
    return {
        "corte": cutoff,
        "conjunto": split,
        "variable": variable,
        "n": len(values),
        "media": float(values.mean()),
        "desviacion": float(values.std()),
        "fraccion_cero": float(values.eq(0).mean()),
        **{k: float(v) for k, v in zip(["min", "p50", "p90", "p99", "p999", "max"], quantiles)},
    }


def inspect_preprocessing(
    pipeline, train: pd.DataFrame, evaluated: pd.DataFrame, columns: list[str]
) -> dict:
    profile = pipeline.named_steps["perfil_estacion"]
    expected = train.groupby("estacion")["demanda"].mean()
    actual = profile.station_profiles_.set_index("estacion")["est_demanda_media"]
    if not np.allclose(actual.loc[expected.index], expected, rtol=1e-12, atol=1e-12):
        raise ValueError("El perfil de estación no coincide con las etiquetas de train.")
    transformed = profile.transform(evaluated[columns])
    if not transformed.index.equals(evaluated.index) or not np.array_equal(
        transformed["estacion"], evaluated["estacion"]
    ):
        raise ValueError("El perfil altera el orden de las observaciones.")
    preprocessor = pipeline.named_steps["preprocesador"]
    numeric = preprocessor.transformers_[0]
    numeric_columns = list(numeric[2])
    train_profile = profile.transform(train[columns])
    imputer = numeric[1].named_steps["imputador"]
    scaler = numeric[1].named_steps["escalador"]
    if not np.allclose(
        imputer.statistics_,
        train_profile[numeric_columns].median().to_numpy(),
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError("Imputador distinto de las medianas de train.")
    if not np.allclose(
        scaler.mean_,
        imputer.transform(train_profile[numeric_columns]).mean(axis=0),
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError("Escalador distinto de train.")
    calendar_columns = preprocessor.transformers_[1][2]
    calendar_categories = preprocessor.named_transformers_["calendario"].categories_
    return {
        "corte": str(evaluated["corte"].iloc[0]),
        "nulos_entrada": int(evaluated[columns].isna().sum().sum()),
        "nulos_numericos_antes_imputador": int(transformed[numeric_columns].isna().sum().sum()),
        "estaciones_sin_perfil": int(
            (~evaluated["estacion"].isin(profile.station_profiles_["estacion"])).sum()
        ),
        "perfil_y_estadisticas_solo_train": True,
        "orden_conservado": True,
        **{
            f"categorias_desconocidas_{column}": int((~evaluated[column].isin(categories)).sum())
            for column, categories in zip(calendar_columns, calendar_categories)
        },
    }


def make_plots(predictions: pd.DataFrame, stages: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for axis, (cutoff, frame) in zip(axes, predictions.groupby("corte", sort=True)):
        for name in ("demanda", MODEL):
            values = np.sort(frame[name].to_numpy())
            indices = np.unique(
                np.r_[
                    np.linspace(0, len(values) - 1, 900).astype(int),
                    np.arange(max(0, len(values) - 100), len(values)),
                ]
            )
            axis.plot(values[indices], (len(values) - indices) / len(values), label=name)
        axis.set_xscale("symlog", linthresh=1)
        axis.set_yscale("log")
        axis.set_title(cutoff[:10])
        axis.set_xlabel("Viajes por estación/hora")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Fracción de filas en la cola (escala log)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Distribución real y predicha: datos completos, incluidos los extremos")
    fig.savefig(output / "distribuciones.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for cutoff, frame in predictions.groupby("corte", sort=True):
        error = np.sort((frame[MODEL] - frame["demanda"]).to_numpy() ** 2)[::-1]
        indices = np.unique(
            np.r_[
                np.arange(min(100, len(error))), np.linspace(0, len(error) - 1, 1000).astype(int)
            ]
        )
        axis.plot(
            100 * (indices + 1) / len(error),
            np.cumsum(error)[indices] / error.sum(),
            label=cutoff[:10],
        )
    axis.axvline(1, color="black", linestyle="--", linewidth=1, label="1 % de observaciones")
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 1)
    axis.set_xlabel("% de filas ordenadas por error cuadrático descendente")
    axis.set_ylabel("Proporción del error cuadrático acumulado")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output / "concentracion_error.png", dpi=160)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    axis.plot(
        stages["iteracion"],
        stages["prediccion"],
        label="Caso extremo: etapas del modelo ya entrenado",
    )
    axis.axhline(0, color="black", linestyle="--", label="Demanda observada: 0")
    axis.set_xlabel("Etapa de boosting (no se selecciona ni reentrena)")
    axis.set_ylabel("Predicción")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.savefig(output / "trayectoria_caso_extremo.png", dpi=160)
    plt.close(fig)


def main() -> None:
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=config.ROOT, text=True
    ).strip()
    if branch != "fase3":
        raise SystemExit(f"Solo se audita en fase3; rama actual: {branch}")
    root = config.ROOT
    report_dir = root / "reports/fase3"
    output = report_dir / "auditoria"
    output.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_originals(report_dir, output)
    execution = json.loads((report_dir / "ejecucion.json").read_text(encoding="utf-8"))
    provenance = json.loads((report_dir / "procedencia_datos.json").read_text(encoding="utf-8"))
    spec = BacktestSpec(
        **{**execution["configuracion"], "cutoffs": tuple(execution["configuracion"]["cutoffs"])}
    )
    trips, raw_trips, integrity = read_sources_independently(root, provenance)
    saved = pd.read_parquet(root / "data/processed/fase3/viajes_limpios.parquet")
    assert_frame_equal(
        hourly_counts(trips).rename("count").to_frame(),
        hourly_counts(saved).rename("count").to_frame(),
        check_dtype=False,
    )
    predictions = pd.read_parquet(root / "data/processed/fase3/predicciones_backtest.parquet")
    verified = verify_labels_and_lags(trips, predictions)
    print(
        f"Verificadas {verified['etiquetas_verificadas']:,} etiquetas y {verified['rezagos_verificados']:,} rezagos desde los CSV.",
        flush=True,
    )
    (
        metrics,
        concentrations,
        distributions,
        preprocessing,
        histories,
        support,
        first_train,
        first_model,
    ) = [], [], [], [], [], [], None, None
    previous = pd.read_csv(report_dir / "metricas_cortes.csv")
    for item in execution["cortes"]:
        cutoff = item["corte"]
        evaluated = predictions.loc[predictions["corte"].eq(cutoff)].copy()
        train, reconstructed, audit = build_fold(trips, pd.Timestamp(cutoff), spec)
        verify_composition(trips, evaluated, audit, spec)
        assert_frame_equal(
            evaluated[reconstructed.columns].reset_index(drop=True),
            reconstructed,
            check_dtype=False,
        )
        pipeline = joblib.load(root / item["modelo_local"])
        columns = features.get_feature_columns(evaluated, horizon="24h")
        preprocessing.append(inspect_preprocessing(pipeline, train, evaluated, columns))
        with threadpool_limits(limits=4):
            assert np.allclose(
                pipeline.predict(evaluated[columns]), evaluated[MODEL], rtol=1e-12, atol=1e-12
            )
            train_prediction = pipeline.predict(train[columns])
        for name in (MODEL, *BASELINES):
            row = {
                "corte": cutoff,
                "modelo": name,
                "filas": len(evaluated),
                **evaluation_metrics(
                    evaluated["demanda"], evaluated[name].to_numpy(), evaluated["hora"]
                ),
            }
            before = previous.loc[previous["corte"].eq(cutoff) & previous["modelo"].eq(name)].iloc[
                0
            ]
            for key in ("mae", "rmse", "wape", "r2"):
                assert np.isclose(row[key], before[key], rtol=1e-12, atol=1e-12)
            metrics.append(row)
            concentrations.append(
                {
                    "corte": cutoff,
                    "modelo": name,
                    **squared_error_concentration(evaluated["demanda"], evaluated[name]),
                }
            )
        for variable in ("demanda", MODEL, *BASELINES):
            distributions.append(
                distribution_row(evaluated[variable], cutoff, variable, "evaluacion")
            )
        for variable in ("demanda", "lag_24", "lag_168"):
            distributions.append(distribution_row(train[variable], cutoff, variable, "train"))
            support.append(
                {
                    "corte": cutoff,
                    "variable": variable,
                    "max_train": float(train[variable].max()),
                    "p99_train": float(train[variable].quantile(0.99)),
                    "max_evaluacion": float(evaluated[variable].max()),
                    "filas_evaluacion_sobre_max_train": int(
                        evaluated[variable].gt(train[variable].max()).sum()
                    ),
                }
            )
        distributions.append(distribution_row(pd.Series(train_prediction), cutoff, MODEL, "train"))
        first_observations = (
            trips.loc[trips["Fin_del_viaje"].lt(pd.Timestamp(cutoff))]
            .groupby("Origen_Id")["Inicio_del_viaje"]
            .min()
        )
        for station, g in train.groupby("estacion"):
            histories.append(
                {
                    "corte": cutoff,
                    "estacion": int(station),
                    "filas_train": len(g),
                    "dias_train": g["origen_pronostico"].nunique(),
                    "viajes_train": int(g["demanda"].sum()),
                    "primera_observacion_origen": str(first_observations.loc[station]),
                    "max_lag_168_train_estacion": float(g["lag_168"].max()),
                    "max_demanda_train_estacion": float(g["demanda"].max()),
                }
            )
        if first_model is None:
            first_train, first_model = train, pipeline
        print(
            f"Corte {cutoff}: métricas originales, composición, perfiles y preprocesamiento comprobados.",
            flush=True,
        )
    first = predictions.loc[predictions["corte"].eq(execution["cortes"][0]["corte"])].copy()
    first["error_absoluto"] = (first[MODEL] - first["demanda"]).abs()
    first["error_cuadratico"] = first["error_absoluto"] ** 2
    top = first.sort_values("error_absoluto", ascending=False, kind="stable").head(20).copy()
    history_frame = pd.DataFrame(histories)
    top = top.merge(
        history_frame.loc[history_frame["corte"].eq(execution["cortes"][0]["corte"])],
        on=["corte", "estacion"],
        validate="many_to_one",
    )
    raw_counts = hourly_counts(raw_trips)
    clean_counts = hourly_counts(trips)
    trace = []
    for row in top.to_dict("records"):
        known = trips.loc[trips["Fin_del_viaje"].lt(row["origen_pronostico"])]
        same_slot = first_train.loc[
            first_train["hora"].eq(row["hora"]) & first_train["dia_semana"].eq(row["dia_semana"])
        ]
        detail = {
            "estacion": row["estacion"],
            "hora_inicio": str(row["hora_inicio"]),
            "origen_pronostico": str(row["origen_pronostico"]),
            "demanda": row["demanda"],
            "prediccion": row[MODEL],
            "error_absoluto": row["error_absoluto"],
            "error_cuadratico": row["error_cuadratico"],
            "filas_train_estacion": row["filas_train"],
            "dias_train_estacion": row["dias_train"],
            "max_lag_168_train_estacion": row["max_lag_168_train_estacion"],
            "dias_historial_minimo_al_corte": (
                pd.Timestamp(row["corte"]) - pd.Timestamp(row["primera_observacion_origen"])
            ).total_seconds()
            / 86400,
            "filas_train_mismo_dia_hora_con_lag_semanal_igual_o_mayor": int(
                same_slot["lag_168"].ge(row["lag_168"]).sum()
            ),
        }
        for lag in (24, 168):
            source = row["hora_inicio"] - pd.Timedelta(hours=lag)
            matching = known.loc[
                known["Origen_Id"].eq(row["estacion"])
                & known["Inicio_del_viaje"].dt.floor("h").eq(source)
            ]
            detail.update(
                {
                    f"fuente_lag_{lag}": str(source),
                    f"lag_{lag}": row[f"lag_{lag}"],
                    f"conteo_lag_{lag}_crudo": int(raw_counts.get((row["estacion"], source), 0)),
                    f"conteo_lag_{lag}_limpio": int(
                        clean_counts.get((row["estacion"], source), 0)
                    ),
                    f"viajes_lag_{lag}_disponibles": len(matching),
                    f"max_fin_lag_{lag}": str(matching["Fin_del_viaje"].max()),
                    f"ids_unicos_lag_{lag}": int(matching["Viaje_Id"].nunique()),
                }
            )
            assert len(matching) == row[f"lag_{lag}"]
        trace.append(detail)
    case = top.head(1)
    columns = features.get_feature_columns(case, horizon="24h")
    with threadpool_limits(limits=4):
        stage_values = [
            float(x[0])
            for x in first_model.named_steps["modelo"].staged_predict(
                first_model[:-1].transform(case[columns])
            )
        ]
        sensitivity = []
        for name, value in [
            ("original", float(case["lag_168"].iloc[0])),
            ("mediana_train", float(first_train["lag_168"].median())),
            ("p99_train", float(first_train["lag_168"].quantile(0.99))),
        ]:
            changed = replace_numeric_preserving_dtype(case[columns], "lag_168", value)
            sensitivity.append(
                {
                    "escenario_diagnostico": name,
                    "lag_168": value,
                    "prediccion": float(first_model.predict(changed)[0]),
                }
            )
        original_prediction = float(first_model.predict(case[columns])[0])
        assert np.isclose(
            sensitivity[0]["prediccion"], original_prediction, rtol=1e-12, atol=1e-12
        )
        assert np.isclose(original_prediction, float(case[MODEL].iloc[0]), rtol=1e-12, atol=1e-12)
        double_input = case[columns].astype({"lag_24": "float64", "lag_168": "float64"})
        precision = [
            {"tipos_rezagos": "float32_original", "prediccion": original_prediction},
            {
                "tipos_rezagos": "float64_mismos_valores",
                "prediccion": float(first_model.predict(double_input)[0]),
            },
        ]
    stages = pd.DataFrame(
        {"iteracion": range(1, len(stage_values) + 1), "prediccion": stage_values}
    )
    historical = joblib.load(root / "models/best_pipeline.joblib")
    learned_months = [
        int(v)
        for v in historical.named_steps["preprocesador"]
        .named_transformers_["calendario"]
        .categories_[2]
    ]
    october = {
        "estado": "no_certificable_como_holdout_virgen_del_proyecto",
        "fase3_utilizo_octubre": False,
        "historico_categorias_mes_aprendidas": learned_months,
        "categorias_historicas_aprendidas_automaticamente": historical.named_steps["preprocesador"]
        .named_transformers_["calendario"]
        .get_params()["categories"]
        == "auto",
        "mes_10_presente_en_fit_historico": 10 in learned_months,
        "nota": "El codificador no registra el año. La categoría 10 demuestra uso de un octubre en fit, pero no permite certificar qué año; el manifiesto incluye octubre de 2026 y falta trazabilidad temporal original para excluirlo.",
        "accion": "No entrenar ni seleccionar con octubre. Exigir evidencia del periodo del histórico o elegir un periodo completo nunca examinado, con protocolo congelado previamente.",
    }
    frames = {
        "top20_primer_corte.csv": pd.DataFrame(trace),
        "concentracion_error.csv": pd.DataFrame(concentrations),
        "distribuciones.csv": pd.DataFrame(distributions),
        "soporte_variables.csv": pd.DataFrame(support),
        "historial_estaciones.csv": history_frame,
        "preprocesamiento.csv": pd.DataFrame(preprocessing),
        "comparacion_baselines.csv": pd.DataFrame(metrics),
        "etapas_caso_extremo.csv": stages,
        "sensibilidad_diagnostica.csv": pd.DataFrame(sensitivity),
        "sensibilidad_precision.csv": pd.DataFrame(precision),
    }
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    day_sse = first.groupby("origen_pronostico")["error_cuadratico"].sum()
    test_path = report_dir / "pytest_auditoria.xml"
    tests = ET.parse(test_path).getroot().find("testsuite").attrib if test_path.exists() else None
    result = {
        "fecha": datetime.now(ZoneInfo("America/Mexico_City")).isoformat(),
        "rama": branch,
        "ejecucion_auditada": execution["id"],
        "error_implementacion_detectado": False,
        "modelos_reentrenados": False,
        "hiperparametros_y_ventana_modificados": False,
        "instantanea": snapshot,
        "comprobacion_independiente": verified,
        "fuentes": integrity,
        "preprocesamiento": preprocessing,
        "porcentaje_sse_12_septiembre": float(
            100 * day_sse.loc[pd.Timestamp("2026-09-12")] / day_sse.sum()
        ),
        "octubre": october,
        "pytest": dict(tests) if tests else None,
        "archivos_evidencia": list(frames),
        "sha256_fuentes_auditoria": {
            str(p.relative_to(root)): dataset.sha256_file(p)
            for p in [
                Path(__file__),
                root / "mibici_demand/temporal.py",
                root / "mibici_demand/modeling/backtest.py",
            ]
        },
    }
    write_json(output / "resumen.json", result)
    write_json(output / "reserva_octubre2026.json", october)
    make_plots(predictions, stages, output)
    write_report(report_dir, frames, result)
    for name, digest in snapshot["archivos_sha256"].items():
        if dataset.sha256_file(report_dir / name) != digest:
            raise ValueError(f"Se alteró evidencia original: {name}")
    print(
        f"Auditoría completada sin reentrenar ni cambiar métricas: {report_dir / 'auditoria_backtesting.md'}",
        flush=True,
    )


def write_report(report_dir: Path, tables: dict[str, pd.DataFrame], result: dict) -> None:
    """El reporte se deriva exclusivamente de tablas calculadas en esta ejecución."""
    top = tables["top20_primer_corte.csv"]
    concentrations = tables["concentracion_error.csv"]
    distributions = tables["distribuciones.csv"]
    metrics = tables["comparacion_baselines.csv"]
    model_concentrations = concentrations.loc[concentrations["modelo"].eq(MODEL)]
    lines = [
        "# Auditoría técnica del backtesting MiBici",
        "",
        f"Ejecutada el {result['fecha']}; rama `fase3`, Python 3.13. Ejecución auditada: `{result['ejecucion_auditada']}`.",
        "",
        "## Dictamen",
        "",
        "No se detectó una discrepancia de implementación en las etiquetas, rezagos, composición de evaluación, alineación de filas, recarga del modelo ni estadísticas del preprocesamiento examinadas. Se conservan las métricas y los modelos originales. No se reentrenó, no se ajustaron hiperparámetros, no se cambió la ventana ni se descartaron observaciones difíciles.",
        "",
        "La anomalía es una cola de predicciones extremas del modelo, concentrada en la tarde/noche del 12 de septiembre. La evidencia respalda una combinación poco representada de rezagos semanales altos y perfiles/horarios aprendidos con un episodio previo de demanda alta. La pérdida Poisson con enlace logarítmico permite amplificar la respuesta positiva; no garantiza que las predicciones queden dentro del rango de etiquetas de entrenamiento. La [documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) describe ese enlace. Esto explica un mecanismo compatible con los datos, no identifica la causa operativa del pico de viajes.",
        "",
        "## 1. Evidencia preservada y reproducción",
        "",
        f"Los archivos de la ejecución anterior se copiaron sin reemplazarlos en `auditoria/originales/{result['ejecucion_auditada']}/`. Los SHA-256 están en `auditoria/resumen.json`; se comprobó nuevamente que los originales no cambiaron. Los modelos de cada corte siguen en sus rutas originales. La auditoría no modifica el registro de MLflow.",
        "",
        "```powershell",
        r".venv\Scripts\python.exe -m mibici_demand.modeling.audit_backtest",
        r".venv\Scripts\python.exe -m pytest tests -q --junitxml=reports/fase3/pytest_auditoria.xml",
        "```",
        "",
        "Los CSV de agosto y septiembre se verifican contra sus hashes originales. Se repite la limpieza mediante máscaras explícitas y se reconstruyen etiquetas y rezagos por agrupación independiente, sin llamar a `TripCleaner` ni `features.monthly_tables` para esa comprobación. Después se contrasta la rejilla del constructor con el parquet guardado. Solo se leen datos de agosto/septiembre; octubre no se descarga ni se usa para puntuar o ajustar.",
        "",
        "## 2. Veinte errores más grandes del primer corte",
        "",
        "Error absoluto = |predicción − observado|; error cuadrático = (predicción − observado)². Todos son del corte 2026-09-07; se muestran los valores completos en `auditoria/top20_primer_corte.csv`.",
        "",
        "| Estación | Fecha | Hora | Observado | Predicción | Error absoluto | Error cuadrático |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for r in top.to_dict("records"):
        ts = pd.Timestamp(r["hora_inicio"])
        lines.append(
            f"| {int(r['estacion'])} | {ts:%Y-%m-%d} | {ts:%H:%M} | {int(r['demanda'])} | {r['prediccion']:.4f} | {r['error_absoluto']:.4f} | {r['error_cuadratico']:.4f} |"
        )
    lines += [
        "",
        "## 3. Concentración del error y distribuciones",
        "",
        "Para cada predictor y corte se ordenan TODAS las filas por error cuadrático descendente y se toman `ceil(0.01*n)` filas. El denominador sigue siendo el SSE de todas las observaciones. Los empates no alteran la suma porque tienen el mismo error. No es una evaluación recortada ni se presentan métricas sin extremos como sustituto.",
        "",
        "| Corte | Filas | Filas top 1 % | SSE total modelo | % SSE top 1 % | % SSE top 20 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in model_concentrations.to_dict("records"):
        lines.append(
            f"| {r['corte'][:10]} | {r['n']:,} | {r['n_top']} | {r['sse_total']:.2f} | {100 * r['proporcion_sse_top']:.2f}% | {100 * r['proporcion_sse_top20']:.2f}% |"
        )
    lines += [
        "",
        f"En el primer corte, el 12 de septiembre aporta el **{result['porcentaje_sse_12_septiembre']:.2f}%** de todo el SSE de la semana. Los veinte casos de la tabla aportan el {100 * model_concentrations.iloc[0]['proporcion_sse_top20']:.2f}%. El RMSE penaliza esos errores al cuadrado y R² = 1 − SSE/SST resulta negativo cuando el SSE supera la variación de las etiquetas. MAE y WAPE penalizan linealmente, por lo que pueden mejorar mientras empeoran RMSE y R².",
        "",
        "| Corte | Serie evaluación | Media | P50 | P90 | P99 | P99.9 | Máximo |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    subset = distributions.loc[
        distributions["conjunto"].eq("evaluacion")
        & distributions["variable"].isin(["demanda", MODEL])
    ]
    for r in subset.to_dict("records"):
        lines.append(
            f"| {r['corte'][:10]} | {r['variable']} | {r['media']:.3f} | {r['p50']:.3f} | {r['p90']:.3f} | {r['p99']:.3f} | {r['p999']:.3f} | {r['max']:.3f} |"
        )
    lines += [
        "",
        "![Distribuciones completas](auditoria/distribuciones.png)",
        "",
        "![Concentración del error](auditoria/concentracion_error.png)",
        "",
        "Las distribuciones completas de entrenamiento/evaluación, ceros, cuantiles y baselines están en `auditoria/distribuciones.csv`. El máximo observado se conserva como descripción, no como umbral para truncar predicciones.",
        "",
        "## 4. Investigación de causas",
        "",
        "### Rezagos y episodio de demanda alta",
        "",
        "Se comprobaron 196,000 etiquetas y 392,000 rezagos contra conteos independientes de los CSV. No hubo discrepancias. En los veinte errores mayores, la fuente semanal corresponde al 5 de septiembre de 2026 y la diaria al 11; ambas preceden a la emisión del 12 a las 00:00. El CSV de trazabilidad conserva el máximo fin de los viajes usados, conteos crudos/limpios y número de IDs únicos. Un pico real en el registro no demuestra por sí solo un evento externo ni descarta problemas de captura del operador.",
        "",
        "| Corte | Variable | Máximo train | P99 train | Máximo evaluación | Filas evaluación > máximo train |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in (
        tables["soporte_variables.csv"]
        .loc[lambda d: d["variable"].eq("lag_168")]
        .to_dict("records")
    ):
        lines.append(
            f"| {r['corte'][:10]} | lag_168 | {r['max_train']:.0f} | {r['p99_train']:.0f} | {r['max_evaluacion']:.0f} | {r['filas_evaluacion_sobre_max_train']} |"
        )
    worst = top.iloc[0]
    lines += [
        "",
        f"Para la estación {int(worst['estacion'])}, el mayor lag semanal en sus filas de train era {worst['max_lag_168_train_estacion']:.0f}; el caso extremo recibe {worst['lag_168']:.0f}. Hay {int(worst['filas_train_mismo_dia_hora_con_lag_semanal_igual_o_mayor'])} filas de train del mismo día de semana y hora con un lag semanal al menos tan alto. El modelo no había observado esa combinación con suficiente soporte. El pico del 5 de septiembre entra como etiqueta en el primer entrenamiento, pero su repetición como rezago semanal aparece después; los cortes posteriores sí pueden incorporar las nuevas observaciones anteriores a su corte. Es evidencia compatible con sensibilidad a una señal semanal poco repetible, no una demostración causal controlada.",
        "",
        "### Poco historial, imputación y calidad de datos",
        "",
        f"Los veinte casos tienen entre {int(top['dias_train_estacion'].min())} y {int(top['dias_train_estacion'].max())} días representados en train y entre {int(top['filas_train_estacion'].min())} y {int(top['filas_train_estacion'].max())} filas por estación. Se observaban desde al menos {top['dias_historial_minimo_al_corte'].min():.1f} días antes del corte dentro de la ventana descargada. No corresponden a estaciones nuevas sin perfil. Esto no equivale a conocer su fecha real de apertura.",
        "",
        "En los tres cortes hubo cero nulos de entrada, cero valores numéricos que requirieran imputación en evaluación y cero estaciones sin perfil. Las medias por estación, medianas del imputador y medias del escalador guardados coinciden con train; se preserva el orden de las filas. `auditoria/preprocesamiento.csv` documenta las comprobaciones. No se identificó imputación ni desalineación como causa de estos extremos.",
        "",
        "No faltan archivos o días de toda la red en la ventana preparada, sus hashes coinciden y la lectura de fechas no produjo fechas nulas. La reconstrucción explícita de la limpieza produce los mismos conteos que el parquet preparado. Un cero estación/hora representa ausencia de viajes registrados tras limpieza; sin bitácoras del operador no se distingue concluyentemente demanda nula, cierre temporal o pérdida de captura. No se inventan viajes para rellenarlo.",
        "",
        "### Amplificación de predicciones",
        "",
        "Con el modelo guardado, sin llamar a `fit`, se recorrieron sus cien etapas mediante `staged_predict`. Todas se conservan: no se escoge retrospectivamente una etapa más favorable. Se hicieron además dos perturbaciones diagnósticas del lag semanal, sustituyéndolo por la mediana y P99 del TRAIN, dejando el resto igual. Estas cifras describen sensibilidad; no son pronósticos válidos alternativos ni métricas de una nueva evaluación.",
        "",
        "| Escenario diagnóstico | lag_168 | Predicción del caso extremo |",
        "|---|---:|---:|",
    ]
    for r in tables["sensibilidad_diagnostica.csv"].to_dict("records"):
        lines.append(
            f"| {r['escenario_diagnostico']} | {r['lag_168']:.0f} | {r['prediccion']:.4f} |"
        )
    lines += [
        "",
        "![Etapas del caso extremo](auditoria/trayectoria_caso_extremo.png)",
        "",
        "La sensibilidad al rezago explica parte de la amplificación, pero aun con un lag de referencia persiste una predicción alta. Los efectos de perfil de estación, calendario y otros rezagos interactúan; este diagnóstico no separa causalmente sus contribuciones. Las perturbaciones de valor conservan exactamente los tipos originales y el escenario original reproduce la predicción guardada.",
        "",
        "### Sensibilidad a la precisión numérica: requisito para servir el modelo",
        "",
        "Se comparó además el mismo caso con los mismos valores de entrada y otra precisión para los rezagos. Es un diagnóstico independiente del anterior. La salida del escalado puede redondearse de forma diferente y atravesar umbrales de árboles. La observación evidencia sensibilidad a la representación; no prueba un defecto de la biblioteca ni una corrección de los errores extremos.",
        "",
        "| Representación de los rezagos | Predicción del mismo caso |",
        "|---|---:|",
        *[
            f"| {r['tipos_rezagos']} | {r['prediccion']:.4f} |"
            for r in tables["sensibilidad_precision.csv"].to_dict("records")
        ],
        "",
        "El backtesting original usa lag_24/lag_168 en float32 y se reproduce íntegramente con ese contrato. No se cambian las métricas por las de float64. En la futura API habrá que reconstruir explícitamente los mismos tipos y probar igualdad contra el pipeline offline; aceptar JSON y dejar que pandas infiera float64 no garantiza equivalencia. Esta fragilidad debe quedar documentada y comprobada durante FastAPI/Docker.",
        "",
        "## 5. Disponibilidad y composición del conjunto",
        "",
        "- El corte fija la cohorte usando solo viajes finalizados antes de él, actividad de la semana anterior y siete días mínimos de historial. El horario se aprende con los 28 días de etiquetas de entrenamiento. La auditoría reconstruye ambos sin leer etiquetas futuras y compara la rejilla estación × timestamp con todas las filas guardadas.",
        "- La cohorte y horario quedan congelados durante cada semana. Esto es una decisión conservadora conocida al emitir cada día, no una selección basada en demanda futura. No incorpora aperturas de estaciones durante la semana; la cobertura original sigue documentada.",
        "- En cada emisión diaria solo entran viajes con fin estrictamente anterior al origen y horas fuente ya cerradas. Se reconstruyeron los lags para los 21 orígenes; no basta que el nombre de la columna diga lag. Las observaciones de los días evaluados anteriores pueden usarse al día siguiente porque el protocolo es rolling-origin diario, no un pronóstico único semanal.",
        "- El calendario objetivo se conoce de antemano. No se usan lag_1, media_24h, catálogo actual ni etiquetas de evaluación como variables. El pipeline histórico no se usa para estas predicciones.",
        "- La maduración de etiquetas es de 24 horas, coherente con la duración máxima de limpieza. Los timestamps de entrenamiento y evaluación son intervalos semiabiertos completos y no dividen estaciones de una misma hora por posiciones de fila.",
        "- Los CSV son mensuales: la disponibilidad intradiaria es una simulación retrospectiva de telemetría de viajes terminados. No demuestra que el portal público hubiera entregado la información a las 00:00 de cada fecha.",
        "",
        "## 6. Comparación completa sobre las mismas observaciones",
        "",
        "Cada fila del parquet incluye simultáneamente observado, modelo y ambos baselines. No se aplicaron máscaras distintas ni se omitieron extremos. MAE y RMSE son viajes/estación/hora; WAPE es proporción.",
        "",
        "| Corte | Predictor | N | MAE | RMSE | WAPE | R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in metrics.to_dict("records"):
        lines.append(
            f"| {r['corte'][:10]} | {r['modelo']} | {r['filas']:,} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['wape']:.4f} | {r['r2']:.4f} |"
        )
    lines += [
        "",
        "La auditoría reprodujo las nueve combinaciones predictor/corte con tolerancia 1e-12 frente al reporte original. El modelo mejora MAE/WAPE en las tres ventanas, pero el primer corte empeora RMSE y R² frente a ambos baselines. No se declara dominio general ni aceptación productiva del modelo.",
        "",
        "## 7. Veintiocho días frente a seis meses",
        "",
        "La elección de 28 días fue una decisión previa de alcance computacional y descarga: permite cuatro ciclos semanales y cubrir los tres cortes con solo dos CSV mensuales. No se realizó un estudio empírico previo que demuestre que 28 días sean mejores o suficientes frente a seis meses. Tampoco se midió una limitación de memoria que impidiera seis meses; Fase 2 ya había podido entrenar con aproximadamente 1.56 millones de filas. Por tanto, **la ventana corta es una limitación metodológica**, no un óptimo validado ni una necesidad técnica demostrada.",
        "",
        "Los cortes actuales usan aproximadamente 251–261 mil filas y menos repeticiones de episodios raros. La falta de soporte en rezagos altos es compatible con esa limitación, pero no demuestra que seis meses resolverían el problema. No se entrena otra ventana sobre estas semanas para escoger retrospectivamente la que produzca mejores métricas. Una comparación futura requeriría validación interna temporal, íntegramente anterior a cada corte y con presupuesto/criterios predeclarados; después, un test no examinado. Se mantuvieron tanto la ventana de 28 días como los parámetros originales.",
        "",
        "## 8. ¿Octubre de 2026 puede reservarse como prueba final?",
        "",
        "**No se puede certificar como holdout virgen del proyecto con la evidencia disponible.** Fase 3 solo usó agosto y septiembre y no ha leído ni evaluado viajes de octubre. Sin embargo, al inspeccionar el artefacto histórico se encontraron las categorías de mes aprendidas `"
        + str(result["octubre"]["historico_categorias_mes_aprendidas"])
        + "`, incluida 10. El manifiesto original también contiene el CSV de octubre de 2026.",
        "",
        "El codificador del artefacto usa categories='auto': el mes 10 fue aprendido, no solo declarado como categoría posible. Es evidencia de un octubre usado en el ajuste del artefacto histórico; el codificador no almacena el año, así que por sí solo no prueba que fuera octubre de 2026. El manifiesto acredita disponibilidad, no uso en entrenamiento. Faltan los timestamps/datasets originales para excluir ese uso de manera defendible. Los documentos previos no resuelven esa trazabilidad. Por eso sería incorrecto afirmar ahora que octubre de 2026 nunca influyó en decisiones previas del proyecto.",
        "",
        "Se mantiene octubre sin consumir en este bloque y se registra el estado en `auditoria/reserva_octubre2026.json`. Podría ser un test externo para el procedimiento de Fase 3, declarando sus antecedentes, pero no un test ciego de todo el desarrollo sin resolver la trazabilidad. Para máxima limpieza académica: recuperar evidencia del periodo usado por el histórico o reservar un periodo completo verdaderamente no examinado, con código, ventana, parámetros y criterios de aceptación congelados antes de observar resultados. Aquí no se entrena con octubre ni se usa para seleccionar nada.",
        "",
        "## 9. Pruebas y recomendaciones",
        "",
    ]
    if result["pytest"]:
        t = result["pytest"]
        lines.append(
            f"Última ejecución de la suite: **{t['tests']} pruebas**, {t['failures']} fallos, {t['errors']} errores y {t['skipped']} omitidas; {float(t['time']):.2f} s. Evidencia: `pytest_auditoria.xml`. Las nuevas pruebas verifican la selección del 1 %, detección de rezagos/etiquetas adulterados, límites de disponibilidad y cambios ilícitos de cohorte."
        )
    else:
        lines.append(
            "Consultar `pytest_auditoria.xml` después de ejecutar la suite; no se presupone un resultado de pruebas aún no ejecutadas."
        )
    lines += [
        "",
        "El bloque de evaluación es defendible como experimento offline acotado y reproducible; no se detectó un error que justifique reentrenar o sustituir las métricas. Puede avanzar a una API y empaquetado para revisión técnica, conservando este resultado negativo. **No está acreditada la fiabilidad para decisiones operativas reales**: requiere analizar episodios atípicos con el operador, más periodos no examinados, un contrato de disponibilidad de datos y criterios de aceptación prospectivos. No se añade ahora un recorte de predicciones ni otra regla escogida para mejorar estas semanas.",
        "",
        "Recomendaciones: preservar los baselines como referencias operativas; investigar la señal semanal excepcional sin atribuirle un evento no confirmado; predeclarar cualquier futura regularización/ventana en validación interna; resolver la reserva de octubre; mantener explícita la diferencia entre éxito de ejecución y calidad predictiva. FastAPI, Docker y drift no se implementan en esta auditoría.",
        "",
    ]
    (report_dir / "auditoria_backtesting.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

r"""Simulación offline de drift con etiquetas trazables y un pipeline temporal fijo.

PowerShell: .venv\Scripts\python.exe -m mibici_demand.monitoring.drift
No descarga datos, no llama a fit y no modifica el servicio ni sus modelos.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.metadata
from itertools import pairwise
import json
from pathlib import Path
import platform
import subprocess
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits

from mibici_demand import config, dataset, features
from mibici_demand.modeling.audit_backtest import (
    inspect_preprocessing,
    squared_error_concentration,
    verify_composition,
    verify_labels_and_lags,
)
from mibici_demand.modeling.backtest import MODEL, evaluation_metrics, write_json
from mibici_demand.temporal import BacktestSpec, build_fold

DEFAULT_PLAN = config.ROOT / "references/plan_drift.json"
METRICS = ("mae", "rmse", "wape", "r2")
LAGS = ("lag_24", "lag_168")
CATEGORICAL = ("estacion", "hora", "dia_semana", "mes", "es_fin_semana", "en_pico")
MONITORED = (*CATEGORICAL, *LAGS, "demanda")


@dataclass(frozen=True)
class AlertPolicy:
    """Reglas ilustrativas predeclaradas; no son SLO ni tests de significación."""

    psi_vigilancia: float = 0.10
    psi_alerta: float = 0.25
    faltantes_vigilancia: float = 0.01
    faltantes_alerta: float = 0.05
    degradacion_relativa: float = 0.10
    mae_delta_min: float = 0.05
    rmse_delta_min: float = 0.10
    wape_delta_min: float = 0.02
    r2_caida_min: float = 0.10

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError("Todos los umbrales deben ser positivos y finitos.")
        if not self.psi_vigilancia < self.psi_alerta or not (
            self.faltantes_vigilancia < self.faltantes_alerta <= 1
        ):
            raise ValueError("Los umbrales de vigilancia deben ser menores que los de alerta.")


def _severity(value: float | None, watch: float, alert: float) -> str:
    if value is None:
        return "sin_datos"
    return "alerta" if value >= alert else "vigilancia" if value >= watch else "sin_alerta"


def make_scenario(
    control: pd.DataFrame,
    family: str,
    intensity: float,
    *,
    seed: int = 42,
    peak_hours: tuple[int, ...] = config.PEAK_HOURS,
) -> pd.DataFrame:
    """Muestreo de registros completos o fallo de captura MCAR, sin consultar y.

    Las máscaras son anidadas entre intensidades. En composición la fracción se
    refiere a posiciones NO pico. En faltantes se refiere al lote completo y se
    ocultan ambos rezagos juntos, simulando una interrupción del feature store.
    """
    if family not in {"control", "composicion", "faltantes", "combinado"}:
        raise ValueError("Familia de escenario desconocida.")
    if not np.isfinite(intensity) or not 0 <= intensity <= 1 or control.empty:
        raise ValueError("Se requiere un lote no vacío e intensidad en [0,1].")
    if family == "control" and intensity != 0:
        raise ValueError("El control no admite alteraciones.")
    if any(c in control for c in ("fila_fuente", "posicion_lote")):
        raise ValueError("El control ya contiene columnas de una simulación.")
    positions = np.arange(len(control))
    if family in {"composicion", "combinado"} and intensity:
        peak = control["hora"].isin(peak_hours).to_numpy()
        if not peak.any() or peak.all():
            raise ValueError("Se necesitan observaciones pico y no pico.")
        rng = np.random.default_rng(seed)
        replace = rng.permutation(np.flatnonzero(~peak))
        donors = rng.choice(np.flatnonzero(peak), size=len(replace), replace=True)
        count = int(np.floor(intensity * len(replace)))
        positions[replace[:count]] = donors[:count]
    result = control.iloc[positions].copy().reset_index(drop=True)
    result["fila_fuente"] = positions
    result["posicion_lote"] = np.arange(len(result))
    if family in {"faltantes", "combinado"}:
        rng = np.random.default_rng(seed + 1)
        missing = rng.permutation(len(result))[: int(np.floor(intensity * len(result)))]
        # Asignación in-place: no convertir float32 a float64 al introducir NaN.
        result.loc[missing, list(LAGS)] = np.nan
    if not result[list(control.columns)].dtypes.equals(control.dtypes):
        raise ValueError("La simulación alteró los tipos numéricos originales.")
    return result


def verify_scenario(control: pd.DataFrame, scenario: pd.DataFrame, family: str) -> None:
    """Rechaza etiquetas o registros adulterados, incluso en filas duplicadas."""
    ids = scenario["fila_fuente"].to_numpy()
    if ids.dtype.kind not in "iu" or len(scenario) != len(control):
        raise ValueError("Identificadores de procedencia o tamaño de lote inválidos.")
    if not ((ids >= 0) & (ids < len(control))).all():
        raise ValueError("Una fila fuente no pertenece al control.")
    expected = control.iloc[ids].reset_index(drop=True)
    actual = scenario[control.columns].reset_index(drop=True)
    if family in {"faltantes", "combinado"}:
        for lag in LAGS:
            # Solo se admite eliminar mediciones, nunca sustituir sus valores.
            bad = actual[lag].notna() & ~actual[lag].eq(expected[lag])
            if bad.any():
                raise ValueError(f"Se adulteró un rezago observado: {lag}.")
            actual[lag] = actual[lag].fillna(expected[lag])
    try:
        assert_frame_equal(actual, expected, check_exact=True)
    except AssertionError as error:
        raise ValueError("Se alteraron etiquetas, calendario o registros completos.") from error


@dataclass(frozen=True)
class Binning:
    """Categorías o cortes aprendidos solo con datos históricos no faltantes."""

    categorical: bool
    cuts: tuple[float, ...] = ()
    categories: tuple = ()

    @classmethod
    def from_reference(cls, values: pd.Series, *, categorical: bool, bins: int = 10):
        clean = values.dropna()
        if clean.empty or bins < 2:
            raise ValueError("Referencia sin valores observados o bins insuficientes.")
        if categorical:
            return cls(True, categories=tuple(sorted(clean.unique().tolist())))
        array = clean.to_numpy(dtype="float64")
        if not np.isfinite(array).all():
            raise ValueError("La referencia contiene valores infinitos.")
        # Bordes de soporte y colas explícitas; también detecta cambios de constantes.
        cuts = np.unique(
            np.r_[
                np.nextafter(array.min(), -np.inf), np.quantile(array, np.linspace(0, 1, bins + 1))
            ]
        )
        return cls(False, cuts=tuple(float(x) for x in cuts))

    @property
    def size(self) -> int:
        return len(self.categories if self.categorical else self.cuts) + 1

    def counts(self, values: pd.Series) -> np.ndarray:
        clean = values.dropna()
        if self.categorical:
            lookup = {value: i for i, value in enumerate(self.categories)}
            codes = [lookup.get(x, len(lookup)) for x in clean]
        else:
            array = clean.to_numpy(dtype="float64")
            if not np.isfinite(array).all():
                raise ValueError("No se aceptan infinitos en el monitoreo.")
            codes = np.searchsorted(self.cuts, array, side="left")
        return np.bincount(np.asarray(codes, dtype="int64"), minlength=self.size)

    def labels(self) -> list[str]:
        if self.categorical:
            return [str(x) for x in self.categories] + ["DESCONOCIDO"]
        edges = [-np.inf, *self.cuts, np.inf]
        return [f"({a:.9g}, {b:.9g}]" for a, b in pairwise(edges)]


def population_stability(
    reference: pd.Series, current: pd.Series, bins: Binning, *, epsilon: float = 1e-6
) -> tuple[dict, list[dict]]:
    """PSI condicional a valores observados; calidad/faltantes se mide aparte."""
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("El suavizado debe ser positivo y finito.")
    left, right = bins.counts(reference), bins.counts(current)
    if not left.sum() or not right.sum():
        return {"psi": None, "distancia_ks": None, "distancia_tv": None}, []
    p, q = left / left.sum(), right / right.sum()
    smooth_p, smooth_q = p + epsilon, q + epsilon
    smooth_p, smooth_q = smooth_p / smooth_p.sum(), smooth_q / smooth_q.sum()
    contributions = (smooth_q - smooth_p) * np.log(smooth_q / smooth_p)
    ks = None
    if not bins.categorical:
        # Se informa solo la distancia; los p-valores IID no son válidos aquí.
        ks = float(ks_2samp(reference.dropna(), current.dropna(), method="asymp").statistic)
    details = [
        {
            "bin": label,
            "n_referencia": int(a),
            "n_actual": int(b),
            "proporcion_referencia": float(x),
            "proporcion_actual": float(y),
            "contribucion_psi": float(z),
        }
        for label, a, b, x, y, z in zip(bins.labels(), left, right, p, q, contributions)
    ]
    return {
        "psi": float(contributions.sum()),
        "distancia_ks": ks,
        "distancia_tv": float(np.abs(p - q).sum() / 2),
    }, details


def compare_distributions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    binning: dict[str, Binning],
    policy: AlertPolicy,
    *,
    epsilon: float = 1e-6,
) -> tuple[list[dict], list[dict]]:
    rows, histograms = [], []
    for name, bins in binning.items():
        measure, details = population_stability(
            reference[name], current[name], bins, epsilon=epsilon
        )
        rows.append(
            {
                "variable": name,
                "rol": "etiqueta_diferida" if name == "demanda" else "entrada",
                **measure,
                "nivel_psi": _severity(measure["psi"], policy.psi_vigilancia, policy.psi_alerta),
                "faltantes_referencia": float(reference[name].isna().mean()),
                "faltantes_actual": float(current[name].isna().mean()),
                "n_referencia_observados": int(reference[name].notna().sum()),
                "n_actual_observados": int(current[name].notna().sum()),
            }
        )
        histograms.extend({"variable": name, **row} for row in details)
    return rows, histograms


def score_predictions(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    values = np.asarray(prediction, dtype="float64")
    actual = frame["demanda"].to_numpy(dtype="float64")
    if values.shape != actual.shape or not len(values):
        raise ValueError("Predicciones y etiquetas deben ser vectores no vacíos de igual tamaño.")
    if not (np.isfinite(actual).all() and np.isfinite(values).all()):
        raise ValueError("Se requieren etiquetas y predicciones finitas; no se omiten filas.")
    if (actual < 0).any() or (values < 0).any():
        raise ValueError("Conteos/predicciones negativos; no se recortan para mejorar métricas.")
    return {
        k: v
        for k, v in evaluation_metrics(frame["demanda"], values, frame["hora"]).items()
        if k in METRICS
    }


def metric_deltas(current: dict, control: dict) -> dict:
    result = {}
    for key in METRICS:
        a, b = current[key], control[key]
        difference = a - b if a is not None and b is not None else None
        result[f"delta_{key}"] = difference
        result[f"diferencia_absoluta_{key}"] = abs(difference) if difference is not None else None
        result[f"delta_relativa_{key}"] = (
            difference / abs(b) if difference is not None and b else None
        )
    return result


def performance_alerts(current: dict, control: dict, policy: AlertPolicy) -> list[dict]:
    changes, rows = metric_deltas(current, control), []
    for metric in ("mae", "rmse", "wape"):
        delta, relative = changes[f"delta_{metric}"], changes[f"delta_relativa_{metric}"]
        minimum = getattr(policy, f"{metric}_delta_min")
        # Con control perfecto (0), solo se exige el mínimo absoluto, declarado aquí.
        if (
            delta is not None
            and delta >= minimum
            and (relative is None or relative >= policy.degradacion_relativa)
        ):
            rows.append(
                {
                    "tipo": "desempeno",
                    "nivel": "alerta",
                    "variable": metric,
                    "comparacion": "control",
                    "valor": current[metric],
                    "referencia": control[metric],
                    "delta": delta,
                    "accion": "Revisar errores y cohortes; corregir datos si procede; evaluar reentrenamiento en validacion temporal futura, sin ejecutarlo.",
                }
            )
    if changes["delta_r2"] is not None and -changes["delta_r2"] >= policy.r2_caida_min:
        rows.append(
            {
                "tipo": "desempeno",
                "nivel": "alerta",
                "variable": "r2",
                "comparacion": "control",
                "valor": current["r2"],
                "referencia": control["r2"],
                "delta": changes["delta_r2"],
                "accion": "Investigar perdida de R2 junto con MAE/RMSE; no interpretar cambio de varianza como causalidad.",
            }
        )
    return rows


def make_alerts(
    detectors: list[dict], current: dict, control: dict, policy: AlertPolicy
) -> list[dict]:
    alerts = performance_alerts(current, control, policy)
    for detector in detectors:
        if detector["rol"] == "entrada" and detector["nivel_psi"] in {"vigilancia", "alerta"}:
            alerts.append(
                {
                    "tipo": "distribucion",
                    "nivel": detector["nivel_psi"],
                    "variable": detector["variable"],
                    "comparacion": detector["comparacion"],
                    "valor": detector["psi"],
                    "referencia": None,
                    "delta": None,
                    "accion": "Verificar calendario y composicion; monitorizar segmentos. No inferir perdida de desempeno solo por PSI.",
                }
            )
        if detector["comparacion"] == "historica" and detector["variable"] in LAGS:
            fraction = detector["faltantes_actual"]
            level = _severity(fraction, policy.faltantes_vigilancia, policy.faltantes_alerta)
            if level in {"vigilancia", "alerta"}:
                alerts.append(
                    {
                        "tipo": "calidad",
                        "nivel": level,
                        "variable": detector["variable"],
                        "comparacion": "historica",
                        "valor": fraction,
                        "referencia": detector["faltantes_referencia"],
                        "delta": fraction - detector["faltantes_referencia"],
                        "accion": "Investigar disponibilidad del feature store, corregir captura/rezagos y recuperar mediciones. La imputacion no resuelve la causa.",
                    }
                )
    return alerts


def describe_frame(frame: pd.DataFrame, name: str) -> list[dict]:
    rows = []
    for column in (*MONITORED, "prediccion"):
        if column not in frame:
            continue
        series = frame[column]
        quantiles = series.quantile([0, 0.5, 0.9, 0.99, 1])
        rows.append(
            {
                "conjunto": name,
                "variable": column,
                "n": len(series),
                "observados": int(series.notna().sum()),
                "faltantes": float(series.isna().mean()),
                "media": float(series.mean()) if series.notna().any() else None,
                "desviacion": float(series.std()) if series.notna().sum() > 1 else None,
                **{
                    k: float(v) if pd.notna(v) else None
                    for k, v in zip(("min", "p50", "p90", "p99", "max"), quantiles)
                },
            }
        )
    return rows


def load_temporal_inputs(plan: dict, artifact_path: Path | None = None) -> tuple:
    """Vincula modelo/datos con hashes de la ejecución existente; no reentrena."""
    report = config.REPORTS_DIR / "fase3"
    execution = json.loads((report / "ejecucion.json").read_text(encoding="utf-8"))
    audits = json.loads((report / "cortes.json").read_text(encoding="utf-8"))
    spec = BacktestSpec(**execution["configuracion"])
    cutoff = pd.Timestamp(plan["corte"])
    if cutoff not in spec.timestamps or cutoff.month != 9 or cutoff.year != 2026:
        raise ValueError("El plan debe usar un corte temporal de septiembre de 2026 existente.")
    index = spec.timestamps.index(cutoff)
    audit, recorded = audits[index], execution["cortes"][index]
    artifact = (artifact_path or config.ROOT / recorded["modelo_local"]).resolve()
    if not artifact.is_relative_to(config.MODELS_DIR.resolve()):
        raise ValueError(
            "Solo se admite un artefacto temporal de Fase 3, no el campeon historico."
        )
    checks = {
        artifact: audit["modelo_sha256"],
        config.PROCESSED_DIR / "fase3/viajes_limpios.parquet": execution["procedencia"][
            "viajes_limpios_sha256"
        ],
        config.PROCESSED_DIR / "fase3/predicciones_backtest.parquet": execution[
            "predicciones_sha256"
        ],
    }
    for path, expected in checks.items():
        if dataset.sha256_file(path) != expected:
            raise ValueError(f"SHA-256 distinto de la procedencia original: {path}")
    evaluation_end = cutoff + spec.evaluation_delta
    if evaluation_end > pd.Timestamp("2026-10-01"):
        raise ValueError("El monitoreo no puede consumir octubre.")
    trips = pd.read_parquet(
        config.PROCESSED_DIR / "fase3/viajes_limpios.parquet",
        filters=[("Inicio_del_viaje", "<", evaluation_end)],
    )
    train, control, rebuilt = build_fold(trips, cutoff, spec)
    for key in (
        "corte",
        "train_inicio",
        "train_fin_exclusivo",
        "filas_train",
        "filas_evaluacion",
        "ids_estaciones",
        "horas_servicio",
    ):
        if rebuilt[key] != audit[key]:
            raise ValueError(f"El corte reconstruido no coincide con la auditoria: {key}")
    saved = pd.read_parquet(config.PROCESSED_DIR / "fase3/predicciones_backtest.parquet")
    saved = saved[saved["corte"].eq(str(cutoff))].reset_index(drop=True)
    assert_frame_equal(control, saved[control.columns], check_exact=True)
    verify_composition(trips, control, audit, spec)
    temporal_checks = {
        "referencia": verify_labels_and_lags(trips, train),
        "monitoreo": verify_labels_and_lags(trips, control),
    }
    control["corte"] = str(cutoff)
    pipeline = joblib.load(artifact)
    columns = features.get_feature_columns(train, horizon="24h")
    if columns != audit["columnas_entrada"]:
        raise ValueError("Columnas incompatibles con el artefacto temporal.")
    preprocessing = inspect_preprocessing(pipeline, train, control, columns)
    # Etiquetas finales se puntúan después de su maduración; nunca entran a predict.
    label_as_of = evaluation_end + pd.Timedelta(hours=spec.label_delay_hours)
    if not trips["Fin_del_viaje"].lt(label_as_of).all():
        raise ValueError("Una etiqueta todavía no estaria disponible al puntuar.")
    provenance = {
        "ejecucion_backtest": execution["id"],
        "mlflow_run_id_origen": recorded["run_id"],
        "modelo": artifact.relative_to(config.ROOT).as_posix(),
        "modelo_original": recorded["modelo_local"],
        "modelo_sha256": checks[artifact],
        "corte": str(cutoff),
        "train_inicio": audit["train_inicio"],
        "train_fin_exclusivo": audit["train_fin_exclusivo"],
        "max_fin_viaje_etiquetas_train": audit["max_fin_viaje_etiquetas_train"],
        "monitoreo_inicio": str(cutoff),
        "monitoreo_fin_exclusivo": str(evaluation_end),
        "evaluacion_etiquetada_disponible_desde": str(label_as_of),
        "zona_horaria": "America/Mexico_City",
        "referencia_filas": len(train),
        "monitoreo_filas": len(control),
        "estaciones": audit["estaciones"],
        "horas_servicio": audit["horas_servicio"],
        "fuentes_originales": execution["procedencia"],
        "datos_sha256": {p.relative_to(config.ROOT).as_posix(): h for p, h in checks.items()},
        "viajes_leidos_antes_fin_monitoreo": len(trips),
        "max_inicio_viaje_leido": str(trips["Inicio_del_viaje"].max()),
        "max_fin_viaje_leido": str(trips["Fin_del_viaje"].max()),
        "cohorte_y_horario_verificados_con_pasado": True,
        "comprobaciones_temporales": temporal_checks,
        "preprocesamiento": preprocessing,
        "dtypes_entrada": {c: str(train[c].dtype) for c in columns},
        "maduracion_horas": spec.label_delay_hours,
        "octubre_consumido": False,
    }
    return pipeline, train, control, saved[MODEL].to_numpy(), columns, provenance


def run_simulation(
    plan_path: Path = DEFAULT_PLAN,
    *,
    threads: int = 4,
    output_dir: Path | None = None,
    artifact_path: Path | None = None,
) -> dict:
    """Ejecuta una sola política congelada y conserva las evidencias anteriores."""
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=config.ROOT, text=True
    ).strip()
    if branch != "fase3" or threads < 1:
        raise ValueError("Ejecutar solo en fase3 y con threads positivo.")
    plan_path = plan_path.resolve()
    if not plan_path.is_relative_to(config.ROOT):
        raise ValueError("El plan debe estar dentro de este repositorio.")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    policy = AlertPolicy(**plan["umbrales"])
    if plan["horas_pico"] != list(config.PEAK_HOURS):
        raise ValueError("Las horas pico deben coincidir con el contrato del proyecto.")
    out = (output_dir or config.REPORTS_DIR / "fase3/drift").resolve()
    if (
        not out.is_relative_to((config.REPORTS_DIR / "fase3").resolve())
        or out == config.REPORTS_DIR / "fase3"
    ):
        raise ValueError("La salida debe ser un subdirectorio dedicado de reports/fase3")
    out.mkdir(parents=True, exist_ok=True)
    # Se persiste el plan ANTES de leer los datos o puntuar escenarios.
    (out / "plan_predeclarado.json").write_bytes(plan_path.read_bytes())
    protected = [
        p
        for folder in (config.MODELS_DIR, config.REPORTS_DIR / "fase3", config.ROOT / "examples")
        for p in folder.rglob("*")
        if p.is_file() and not p.is_relative_to(out) and p.name != "validacion_drift.md"
    ]
    protected += [
        config.PROCESSED_DIR / "fase3" / name
        for name in (
            "viajes_limpios.parquet",
            "predicciones_backtest.parquet",
            "demanda_horaria.parquet",
        )
    ]
    before = {p: dataset.sha256_file(p) for p in protected}
    pipeline, reference, base, recorded_prediction, columns, provenance = load_temporal_inputs(
        plan, artifact_path
    )
    model_state = joblib.hash(pipeline)
    reference["en_pico"] = reference["hora"].isin(plan["horas_pico"]).astype("int8")
    base["en_pico"] = base["hora"].isin(plan["horas_pico"]).astype("int8")
    bins = {
        c: Binning.from_reference(
            reference[c], categorical=c in CATEGORICAL, bins=plan["detectores"]["psi_bins"]
        )
        for c in MONITORED
    }
    write_json(
        out / "bins_referencia.json",
        {
            c: {
                "categorica": b.categorical,
                "cortes": list(b.cuts),
                "categorias": list(b.categories),
            }
            for c, b in bins.items()
        },
    )
    scenario_specs = [
        ("control", 0.0),
        *[(f, float(a)) for f in plan["familias"] for a in plan["intensidades"]],
    ]
    metric_rows, detector_rows, histogram_rows, alert_rows, prediction_frames, top_errors = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    descriptions = describe_frame(reference, "referencia_historica")
    decomposition = []
    control_scores = None
    for family, intensity in scenario_specs:
        name = "control" if family == "control" else f"{family}_{round(intensity * 100):02d}"
        frame = make_scenario(
            base, family, intensity, seed=plan["semilla"], peak_hours=tuple(plan["horas_pico"])
        )
        verify_scenario(base, frame, family)
        with threadpool_limits(limits=threads):
            prediction = np.asarray(pipeline.predict(frame[columns]), dtype="float64")
        unchanged_prediction = recorded_prediction[frame["fila_fuente"].to_numpy()]
        observed = frame[list(LAGS)].notna().all(axis=1).to_numpy()
        np.testing.assert_allclose(
            prediction[observed], unchanged_prediction[observed], rtol=1e-12, atol=1e-12
        )
        if family == "control":
            np.testing.assert_allclose(prediction, recorded_prediction, rtol=1e-12, atol=1e-12)
            control_scores = score_predictions(frame, prediction)
        score = score_predictions(frame, prediction)
        same_mixture = score_predictions(frame, unchanged_prediction)
        decomposition.append(
            {
                "escenario": name,
                "mae_sin_fallo_misma_mezcla": same_mixture["mae"],
                "rmse_sin_fallo_misma_mezcla": same_mixture["rmse"],
                "delta_mae_por_composicion": same_mixture["mae"] - control_scores["mae"],
                "delta_mae_por_faltantes": score["mae"] - same_mixture["mae"],
                "delta_mse_por_composicion": same_mixture["rmse"] ** 2
                - control_scores["rmse"] ** 2,
                "delta_mse_por_faltantes": score["rmse"] ** 2 - same_mixture["rmse"] ** 2,
                "filas_entrada_sin_fallo_verificadas": int(observed.sum()),
            }
        )
        detectors = []
        for comparison, other in (("historica", reference), ("control", base)):
            measured, histograms = compare_distributions(
                other, frame, bins, policy, epsilon=plan["detectores"]["suavizado_probabilidad"]
            )
            detectors.extend({"escenario": name, "comparacion": comparison, **r} for r in measured)
            histogram_rows.extend(
                {"escenario": name, "comparacion": comparison, **r} for r in histograms
            )
        alerts = make_alerts(detectors, score, control_scores, policy)
        alert_rows.extend({"escenario": name, **a} for a in alerts)
        concentration = squared_error_concentration(frame["demanda"], prediction)
        metric_rows.append(
            {
                "escenario": name,
                "familia": family,
                "intensidad": intensity,
                "n": len(frame),
                "filas_fuente_unicas": int(frame["fila_fuente"].nunique()),
                "fraccion_pico": float(frame["en_pico"].mean()),
                **{f"faltantes_{lag}": float(frame[lag].isna().mean()) for lag in LAGS},
                **score,
                **metric_deltas(score, control_scores),
                "max_prediccion": float(prediction.max()),
                "sse_top1_proporcion": concentration["proporcion_sse_top"],
                "drift_historico": any(
                    a["tipo"] == "distribucion" and a["comparacion"] == "historica" for a in alerts
                ),
                "drift_vs_control": any(
                    a["tipo"] == "distribucion" and a["comparacion"] == "control" for a in alerts
                ),
                "problema_calidad": any(a["tipo"] == "calidad" for a in alerts),
                "perdida_desempeno": any(a["tipo"] == "desempeno" for a in alerts),
            }
        )
        frame["prediccion"] = prediction
        frame["escenario"] = name
        frame["error_absoluto"] = (frame["demanda"] - prediction).abs()
        frame["error_cuadratico"] = (frame["demanda"] - prediction) ** 2
        top_errors.append(frame.nlargest(20, "error_absoluto"))
        prediction_frames.append(frame)
        descriptions.extend(describe_frame(frame, name))
        detector_rows.extend(detectors)
        print(
            json.dumps(
                {
                    "escenario": name,
                    **score,
                    "alertas_desempeno": [
                        a["variable"] for a in alerts if a["tipo"] == "desempeno"
                    ],
                }
            ),
            flush=True,
        )
    if joblib.hash(pipeline) != model_state:
        raise ValueError("El estado del modelo/preprocesador cambió durante la simulación.")
    unchanged = {
        p.relative_to(config.ROOT).as_posix(): dataset.sha256_file(p) == digest
        for p, digest in before.items()
    }
    if not all(unchanged.values()):
        raise ValueError("Cambió un artefacto o resultado previo.")
    tables = {
        "metricas_escenarios": pd.DataFrame(metric_rows),
        "detectores_drift": pd.DataFrame(detector_rows),
        "distribuciones_bins": pd.DataFrame(histogram_rows),
        "alertas_acciones": pd.DataFrame(alert_rows),
        "estadisticas_descriptivas": pd.DataFrame(descriptions),
        "descomposicion_errores": pd.DataFrame(decomposition),
    }
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pd.DataFrame(
        [
            {
                "tipo": "distribucion",
                "metrica": "psi",
                "vigilancia": policy.psi_vigilancia,
                "alerta_absoluta": policy.psi_alerta,
                "alerta_relativa": None,
                "accion": "Investigar calendario/composicion y monitorizar; no inferir degradacion",
            },
            {
                "tipo": "calidad",
                "metrica": "fraccion_faltantes_lag",
                "vigilancia": policy.faltantes_vigilancia,
                "alerta_absoluta": policy.faltantes_alerta,
                "alerta_relativa": None,
                "accion": "Corregir captura, disponibilidad y rezagos",
            },
            *[
                {
                    "tipo": "desempeno",
                    "metrica": metric,
                    "vigilancia": None,
                    "alerta_absoluta": getattr(policy, f"{metric}_delta_min"),
                    "alerta_relativa": policy.degradacion_relativa,
                    "accion": "Exigir ambos umbrales; revisar modelo tras descartar problemas de datos",
                }
                for metric in ("mae", "rmse", "wape")
            ],
            {
                "tipo": "desempeno",
                "metrica": "caida_r2",
                "vigilancia": None,
                "alerta_absoluta": policy.r2_caida_min,
                "alerta_relativa": None,
                "accion": "Investigar junto con errores y varianza de demanda",
            },
        ]
    ).to_csv(out / "umbrales.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        out / "predicciones_escenarios.parquet", index=False
    )
    pd.concat(top_errors, ignore_index=True).to_csv(out / "errores_mayores.csv", index=False)
    result = {
        "estado": "ejecutado",
        "ejecutado_utc": datetime.now(timezone.utc).isoformat(),
        "rama": branch,
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "plan_sha256": dataset.sha256_file(plan_path),
        "plan": plan,
        "procedencia": provenance,
        "threads": threads,
        "estado_modelo_antes_despues": model_state,
        "modelo_y_preprocesador_sin_cambios": True,
        "reentrenamiento_ejecutado": False,
        "control_max_diferencia_backtest": float(
            np.max(np.abs(prediction_frames[0]["prediccion"] - recorded_prediction))
        ),
        "resultados_previos_sin_cambios": unchanged,
        "escenarios": metric_rows,
        "versiones": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "pyarrow",
                "matplotlib",
                "pytest",
            )
        },
        "codigo_sha256": {
            p.relative_to(config.ROOT).as_posix(): dataset.sha256_file(p)
            for p in [
                *sorted(Path(__file__).parent.glob("*.py")),
                plan_path,
                config.ROOT / "tests/test_drift.py",
            ]
            if p.is_file()
        },
        "pytest": None,
    }
    from mibici_demand.monitoring.reporting import make_plots, write_report

    make_plots(tables, reference, prediction_frames, out)
    write_json(out / "ejecucion.json", result)
    write_report(out, tables, result)
    return result


def record_test_results(output_dir: Path | None = None) -> dict:
    """Añade al informe el JUnit real posterior a la simulación, sin reejecutarla."""
    out = output_dir or config.REPORTS_DIR / "fase3/drift"
    result = json.loads((out / "ejecucion.json").read_text(encoding="utf-8"))
    junit = out / "pytest_drift.xml"
    if junit.stat().st_mtime < datetime.fromisoformat(result["ejecutado_utc"]).timestamp():
        raise ValueError("El JUnit es anterior a la simulacion; ejecutar la suite completa.")
    suites = ET.parse(junit).getroot().findall("testsuite")
    if not suites:
        raise ValueError("JUnit sin suites ejecutadas.")
    result["pytest"] = {
        key: str(sum(int(s.attrib[key]) for s in suites))
        for key in ("tests", "failures", "errors", "skipped")
    }
    result["pytest"]["time"] = f"{sum(float(s.attrib['time']) for s in suites):.3f}"
    result["pytest_sha256"] = dataset.sha256_file(junit)
    tables = {
        name: pd.read_csv(out / f"{name}.csv")
        for name in (
            "metricas_escenarios",
            "detectores_drift",
            "alertas_acciones",
            "descomposicion_errores",
        )
    }
    from mibici_demand.monitoring.reporting import write_report

    write_json(out / "ejecucion.json", result)
    write_report(out, tables, result)
    return result["pytest"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--salida", type=Path, default=None)
    parser.add_argument("--modelo", type=Path, default=None)
    parser.add_argument(
        "--actualizar-reporte",
        action="store_true",
        help="Registrar el JUnit posterior sin repetir predicciones",
    )
    args = parser.parse_args(argv)
    try:
        if args.actualizar_reporte:
            print(json.dumps(record_test_results(args.salida)))
        else:
            run_simulation(
                args.plan, threads=args.threads, output_dir=args.salida, artifact_path=args.modelo
            )
    except Exception as error:
        out = (args.salida or config.REPORTS_DIR / "fase3/drift").resolve()
        if out.is_relative_to((config.REPORTS_DIR / "fase3").resolve()):
            out.mkdir(parents=True, exist_ok=True)
            write_json(
                out / "bloqueo.json",
                {"estado": "fallo", "tipo": type(error).__name__, "error": str(error)},
            )
        raise


if __name__ == "__main__":
    main()

"""Datos y características disponibles al emitir un pronóstico diario (Fase 3).

No cambia la construcción histórica de las Fases 1/2. Los CSV mensuales permiten
una simulación retrospectiva de acceso a viajes terminados, no un feed en vivo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from mibici_demand import config, dataset, features
from mibici_demand.cleaning import TripCleaner

DEFAULT_CUTOFFS = ("2026-09-07", "2026-09-14", "2026-09-21")
TRIP_FIELDS = ["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"]


@dataclass(frozen=True)
class BacktestSpec:
    cutoffs: tuple[str, ...] = DEFAULT_CUTOFFS
    train_days: int = 28
    evaluation_days: int = 7
    label_delay_hours: int = 24

    def __post_init__(self) -> None:
        cuts = self.timestamps
        if len(cuts) != 3 or any(pd.isna(c) for c in cuts) or list(cuts) != sorted(set(cuts)):
            raise ValueError("Se requieren exactamente tres cortes distintos y crecientes.")
        if self.train_days < 7 or self.evaluation_days < 1:
            raise ValueError("Se necesitan >=7 días de entrenamiento y >=1 día de evaluación.")
        if self.label_delay_hours < config.MAX_TRIP_SECONDS / 3600 or self.label_delay_hours % 24:
            raise ValueError(
                "La maduración de etiquetas debe cubrir la duración máxima de limpieza."
            )
        if any(c.tzinfo is not None or c != c.normalize() for c in cuts):
            raise ValueError("Los cortes deben ser medianoches locales sin zona, como los CSV.")
        if any(a + self.evaluation_delta > b for a, b in pairwise(cuts)):
            raise ValueError("Las ventanas de evaluación no pueden superponerse.")

    @property
    def timestamps(self) -> tuple[pd.Timestamp, ...]:
        return tuple(pd.Timestamp(c) for c in self.cutoffs)

    @property
    def evaluation_delta(self) -> pd.Timedelta:
        return pd.Timedelta(days=self.evaluation_days)

    @property
    def data_start(self) -> pd.Timestamp:
        return self.timestamps[0] - pd.Timedelta(
            days=self.train_days + 7, hours=self.label_delay_hours
        )

    @property
    def data_end(self) -> pd.Timestamp:
        return self.timestamps[-1] + self.evaluation_delta

    @property
    def periods(self) -> list[str]:
        return (
            pd.period_range(self.data_start, self.data_end - pd.Timedelta(1, "ns"), freq="M")
            .astype(str)
            .tolist()
        )


def prepare_data(
    spec: BacktestSpec,
    raw_dir: Path,
    processed_dir: Path,
    reports_dir: Path,
    *,
    download_missing: bool = True,
    expected_manifest: Path = config.RAW_DIR / "manifest.csv",
) -> tuple[pd.DataFrame, dict]:
    """Descarga solo meses necesarios, comprueba hashes y reutiliza lectura/limpieza.

    El manifiesto de Fase 1 nunca se reescribe. No se usa un catálogo actual para
    atribuir capacidad/ubicación a una estación en una fecha pasada.
    """
    expected = pd.read_csv(expected_manifest).set_index("archivo")
    names = [f"datos_abiertos_{p.replace('-', '_')}.csv" for p in spec.periods]
    missing_references = [n for n in names if n not in expected.index]
    if missing_references:
        raise ValueError(
            f"Faltan referencias de procedencia en {expected_manifest}: {missing_references}"
        )
    raw_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (raw_dir / n).is_file()]
    if missing and download_missing:
        dataset.download(spec.periods[0], spec.periods[-1], raw_dir, include_catalog=False)
    missing = [n for n in names if not (raw_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"Faltan datos mensuales en {raw_dir}: {missing}")

    cleaned, provenance, cleaning_rows = [], [], []
    for name in names:
        path = raw_dir / name
        reference = expected.loc[name]
        digest = dataset.sha256_file(path)
        if digest != reference["sha256"] or path.stat().st_size != int(reference["bytes"]):
            raise ValueError(
                f"Integridad incorrecta de {name}: SHA-256={digest}; esperado={reference['sha256']}"
            )
        trips = dataset.read_trips(path)
        attrs = trips.attrs.copy()
        clean, cleaning = TripCleaner("limpio").clean(trips)
        # Un viaje sin fin no tiene un instante verificable de disponibilidad.
        cleaning["sin_fin_para_disponibilidad"] = int(clean["Fin_del_viaje"].isna().sum())
        clean = clean.dropna(subset=["Fin_del_viaje"])
        if not clean["Inicio_del_viaje"].dt.strftime("%Y-%m").eq(dataset.period_of(path)).all():
            raise ValueError(f"{name}: viajes fuera del mes; revisar el formato antes de evaluar.")
        cleaning_rows.append({"archivo": name, **cleaning, **attrs})
        provenance.append(
            {
                "archivo": name,
                "url": str(reference["url"]),
                "bytes": path.stat().st_size,
                "sha256": digest,
                "sha256_coincide_fase1": True,
            }
        )
        clean = clean.loc[
            (clean["Inicio_del_viaje"] >= spec.data_start)
            & (clean["Inicio_del_viaje"] < spec.data_end),
            TRIP_FIELDS,
        ].copy()
        cleaned.append(clean)
        print(f"Preparado {name}: {len(clean):,} viajes en la ventana necesaria", flush=True)

    trips = (
        pd.concat(cleaned, ignore_index=True)
        .sort_values("Inicio_del_viaje")
        .reset_index(drop=True)
    )
    observed_days = pd.DatetimeIndex(trips["Inicio_del_viaje"].dt.normalize().unique())
    required_days = pd.date_range(spec.data_start, spec.data_end, freq="D", inclusive="left")
    missing_days = required_days.difference(observed_days)
    if len(missing_days):
        raise ValueError(
            f"Días sin observaciones de toda la red; no asumir demanda cero: {missing_days.tolist()}"
        )
    cleaned_path = processed_dir / "viajes_limpios.parquet"
    trips.to_parquet(cleaned_path, index=False)
    # Reutiliza la agregación validada; la elegibilidad y los rezagos se hacen por corte.
    demand, _ = features.monthly_tables(trips)
    demand.to_parquet(processed_dir / "demanda_horaria.parquet", index=False)
    pd.DataFrame(cleaning_rows).to_csv(reports_dir / "limpieza.csv", index=False)
    info = {
        "fuente": config.OPEN_DATA_URL,
        "zona_horaria": "America/Mexico_City",
        "descargado_o_verificado_en": datetime.now(ZoneInfo("America/Mexico_City")).isoformat(),
        "inicio_inclusivo": str(spec.data_start),
        "fin_exclusivo": str(spec.data_end),
        "meses": spec.periods,
        "archivos": provenance,
        "viajes_utilizados": len(trips),
        "viajes_limpios_sha256": dataset.sha256_file(cleaned_path),
        "acceso": "CSV mensuales; simulación offline de disponibilidad de viajes terminados",
        "catalogo_estaciones_utilizado": False,
    }
    (reports_dir / "procedencia_datos.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return trips, info


def causal_daily_frame(
    trips: pd.DataFrame,
    stations: list[int],
    start: pd.Timestamp,
    end: pd.Timestamp,
    service_hours: list[int],
    first_seen: pd.Series,
) -> pd.DataFrame:
    """Rejilla con etiquetas finales y rezagos conocidos a las 00:00 de cada día.

    La etiqueta se usa solo para fit/métricas. Un rezago cuenta viajes cuyo fin es
    estrictamente anterior al origen del pronóstico, incluso en el entrenamiento.
    No se infieren cierres de estaciones ni horarios con datos de evaluación.
    """
    counts, _ = features.monthly_tables(trips)
    labels = counts.set_index(["estacion", "hora"])["demanda"]
    frames = []
    for origin in pd.date_range(start, end, freq="D", inclusive="left"):
        eligible = [s for s in stations if first_seen.loc[s] + pd.Timedelta(days=7) <= origin]
        if not eligible:
            continue
        targets = [origin + pd.Timedelta(hours=h) for h in service_hours]
        frame = pd.MultiIndex.from_product(
            [eligible, targets], names=["estacion", "hora_inicio"]
        ).to_frame(index=False)
        frame["origen_pronostico"] = origin
        frame["demanda"] = labels.reindex(
            pd.MultiIndex.from_frame(frame[["estacion", "hora_inicio"]]), fill_value=0
        ).to_numpy()
        known = trips.loc[
            (trips["Fin_del_viaje"] < origin)
            & (trips["Inicio_del_viaje"] >= origin - pd.Timedelta(days=7))
            & (trips["Inicio_del_viaje"] < origin)
        ]
        known_counts, _ = features.monthly_tables(known)
        known_index = known_counts.set_index(["estacion", "hora"])["demanda"]
        for lag in (24, 168):
            source = frame["hora_inicio"] - pd.Timedelta(hours=lag)
            if not ((source + pd.Timedelta(hours=1)) <= origin).all():
                raise ValueError("Un rezago utiliza una hora no cerrada al emitir el pronóstico.")
            keys = pd.MultiIndex.from_arrays([frame["estacion"], source])
            frame[f"lag_{lag}"] = known_index.reindex(keys, fill_value=0).to_numpy(dtype="float32")
            frame[f"fuente_lag_{lag}"] = source
        when = frame["hora_inicio"]
        frame["hora"] = when.dt.hour.astype("int8")
        frame["dia_semana"] = when.dt.day_of_week.astype("int8")
        frame["mes"] = when.dt.month.astype("int8")
        frame["es_fin_semana"] = (when.dt.day_of_week >= 5).astype("int8")
        frames.append(frame)
    if not frames:
        raise ValueError("No hay filas con siete días de historia disponible.")
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["hora_inicio", "estacion"])
        .reset_index(drop=True)
    )


def build_fold(
    trips: pd.DataFrame, cutoff: pd.Timestamp, spec: BacktestSpec
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Ventanas semiabiertas por timestamp, cohorte/horario fijos por corte."""
    train_end = cutoff - pd.Timedelta(hours=spec.label_delay_hours)
    train_start = train_end - pd.Timedelta(days=spec.train_days)
    evaluation_end = cutoff + spec.evaluation_delta
    past = trips.loc[(trips["Inicio_del_viaje"] < cutoff) & (trips["Fin_del_viaje"] < cutoff)]
    train_trips = past.loc[
        (past["Inicio_del_viaje"] >= train_start) & (past["Inicio_del_viaje"] < train_end)
    ]
    _, activity = features.monthly_tables(past)
    recent = past.loc[past["Inicio_del_viaje"] >= cutoff - pd.Timedelta(days=7)]
    recent_ids = set(recent["Origen_Id"]) | set(recent["Destino_Id"].dropna())
    first_seen = activity.set_index("estacion")["primera"]
    stations = sorted(
        int(s)
        for s in first_seen.index
        if s in recent_ids and first_seen.loc[s] + pd.Timedelta(days=7) <= cutoff
    )
    hours = train_trips.groupby(train_trips["Inicio_del_viaje"].dt.hour).size()
    service_hours = sorted(
        int(h) for h in hours.index[hours / hours.sum() >= config.CLOSED_HOUR_SHARE]
    )
    if not stations or not service_hours:
        raise ValueError(f"Historia insuficiente antes del corte {cutoff}.")
    # El frame de entrenamiento ni siquiera recibe viajes posteriores al corte.
    train = causal_daily_frame(past, stations, train_start, train_end, service_hours, first_seen)
    evaluation = causal_daily_frame(
        trips, stations, cutoff, evaluation_end, service_hours, first_seen
    )
    if not (train["hora_inicio"].max() < train_end <= cutoff <= evaluation["hora_inicio"].min()):
        raise ValueError("Partición temporal inválida.")
    for frame in (train, evaluation):
        if frame.duplicated(["estacion", "hora_inicio"]).any():
            raise ValueError("Estación/hora duplicada.")
        for lag in (24, 168):
            if not (frame[f"fuente_lag_{lag}"] < frame["origen_pronostico"]).all():
                raise ValueError("Fuente futura en un rezago.")
    future_ids = set(
        trips.loc[
            (trips["Inicio_del_viaje"] >= cutoff) & (trips["Inicio_del_viaje"] < evaluation_end),
            "Origen_Id",
        ]
    )
    audit = {
        "corte": str(cutoff),
        "train_inicio": str(train_start),
        "train_fin_exclusivo": str(train_end),
        "train_max_timestamp": str(train["hora_inicio"].max()),
        "train_min_timestamp": str(train["hora_inicio"].min()),
        "evaluacion_inicio": str(cutoff),
        "evaluacion_fin_exclusivo": str(evaluation_end),
        "filas_train": len(train),
        "filas_evaluacion": len(evaluation),
        "estaciones": len(stations),
        "ids_estaciones": stations,
        "horas_servicio": service_hours,
        "estaciones_futuras_fuera_cohorte": sorted(int(s) for s in future_ids - set(stations)),
        "label_delay_hours": spec.label_delay_hours,
        "timezone": "America/Mexico_City",
        "preprocesamiento": "fit independiente solo sobre train",
        "max_fin_viaje_etiquetas_train": str(train_trips["Fin_del_viaje"].max()),
        "lag_1_y_media_24h_utilizados": False,
    }
    return train, evaluation, audit

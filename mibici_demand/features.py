"""De viajes limpios a la tabla de entrenamiento: demanda por estación y hora, con variables de calendario y de rezago."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mibici_demand import config

HOUR = pd.Timedelta(hours=1)
LAG_COLUMNS = [f"lag_{k}" for k in config.LAGS] + ["media_24h"]
FEATURES = ["estacion", "hora", "dia_semana", "mes", "es_fin_semana"] + LAG_COLUMNS
TARGET = "demanda"


# ---------------------------------------------------------------------------------------------------------------
# Paso 1: tablas pequeñas por mes (se guardan en data/interim para no cargar todo en memoria)
# ---------------------------------------------------------------------------------------------------------------
def monthly_tables(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (demanda, actividad) de un mes.
    demanda:   viajes iniciados por estación y hora, solo donde hubo viajes.
    actividad: primera y última hora en que cada estación aparece, como origen o como destino."""
    hours = clean["Inicio_del_viaje"].dt.floor("h")
    demanda = (
        clean.assign(hora=hours)
        .groupby(["Origen_Id", "hora"])
        .size()
        .rename("demanda")
        .reset_index()
        .rename(columns={"Origen_Id": "estacion"})
    )
    demanda["estacion"] = demanda["estacion"].astype("int16")
    demanda["demanda"] = demanda["demanda"].astype("int16")

    as_origin = clean.groupby("Origen_Id")["Inicio_del_viaje"].agg(["min", "max"])
    ends = clean.dropna(subset=["Destino_Id", "Fin_del_viaje"])
    as_dest = ends.groupby("Destino_Id")["Fin_del_viaje"].agg(["min", "max"])
    both = pd.concat([as_origin, as_dest])
    activity = both.groupby(level=0).agg(primera=("min", "min"), ultima=("max", "max"))
    activity.index = activity.index.astype("int16")
    activity = activity.rename_axis("estacion").reset_index()
    activity["primera"] = activity["primera"].dt.floor("h")
    activity["ultima"] = activity["ultima"].dt.floor("h")
    return demanda, activity


# ---------------------------------------------------------------------------------------------------------------
# Paso 2: rejilla estación-hora solo donde la estación existía, con rezagos
# ---------------------------------------------------------------------------------------------------------------
def _open_hours(demanda: pd.DataFrame) -> np.ndarray:
    """Claves (año*12+mes)*24+hora de las horas con servicio: las que concentran al menos CLOSED_HOUR_SHARE de los viajes del mes."""
    month = demanda["hora"].dt.year * 12 + demanda["hora"].dt.month
    hod = demanda["hora"].dt.hour
    totals = demanda.groupby([month.rename("mes_id"), hod.rename("hod")])["demanda"].sum()
    share = totals / totals.groupby(level=0).transform("sum")
    keys = [mes_id * 24 + h for mes_id, h in share[share >= config.CLOSED_HOUR_SHARE].index]
    return np.array(keys, dtype=np.int64)


def build_station_hour_table(
    interim_dir: Path, desde: str | None = None, hasta: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Construye la tabla de entrenamiento a partir de las tablas mensuales de data/interim.

    Reglas de la rejilla:
    - Cada estación existe solo entre su primera y su última aparición: antes de abrir o después de cerrar no hay demanda cero,
      no hay demanda posible.
    - Las horas sin servicio (por ejemplo de madrugada) se usan para calcular rezagos como ceros, pero no entran como filas
      de entrenamiento porque no son demanda.
    - Se descartan las filas sin historial suficiente para los rezagos.
    """
    from mibici_demand.dataset import in_window

    d_files = sorted(
        p
        for p in interim_dir.glob("demanda_*.parquet")
        if in_window(p.stem[len("demanda_") :], desde, hasta)
    )
    if not d_files:
        raise FileNotFoundError(f"no hay tablas mensuales en {interim_dir} para esa ventana")
    demanda = pd.concat([pd.read_parquet(p) for p in d_files], ignore_index=True)
    activity = pd.concat(
        [pd.read_parquet(p.with_name(p.name.replace("demanda_", "actividad_"))) for p in d_files],
        ignore_index=True,
    )
    activity = (
        activity.groupby("estacion")
        .agg(primera=("primera", "min"), ultima=("ultima", "max"))
        .reset_index()
        .sort_values("estacion")
    )

    t0 = activity["primera"].min()
    h0 = ((activity["primera"] - t0) // HOUR).to_numpy(np.int64)
    h1 = ((activity["ultima"] - t0) // HOUR).to_numpy(np.int64)
    lengths = h1 - h0 + 1
    stations = activity["estacion"].to_numpy(np.int16)
    n_stations, total = len(stations), int(lengths.sum())

    # Rejilla ordenada por estación y hora
    pos = np.repeat(np.arange(n_stations), lengths)
    offsets = np.cumsum(lengths) - lengths
    hour_idx = (
        np.arange(total, dtype=np.int64) - np.repeat(offsets, lengths) + np.repeat(h0, lengths)
    )
    span = int(h1.max()) + 1
    grid_key = pos.astype(np.int64) * span + hour_idx

    # Demanda observada sobre la rejilla (el resto son ceros reales: hubo servicio y no hubo viajes)
    demand_pos = np.searchsorted(stations, demanda["estacion"].to_numpy())
    demand_hour = ((demanda["hora"] - t0) // HOUR).to_numpy(np.int64)
    demand_key = demand_pos.astype(np.int64) * span + demand_hour
    where = np.searchsorted(grid_key, demand_key)
    assert np.all(grid_key[where] == demand_key), (
        "hay demanda fuera de la ventana activa de su estación"
    )
    values = np.zeros(total, dtype=np.int16)
    values[where] = demanda["demanda"].to_numpy(np.int16)

    table = pd.DataFrame(
        {
            "estacion": np.repeat(stations, lengths),
            "hora_inicio": t0 + pd.to_timedelta(hour_idx, unit="h"),
            TARGET: values,
        }
    )

    # Rezagos y media móvil: solo con información anterior a la hora que se predice
    by_station = table.groupby("estacion", sort=False)[TARGET]
    for k in config.LAGS:
        table[f"lag_{k}"] = by_station.shift(k).astype("float32")
    table["media_24h"] = by_station.transform(
        lambda s: s.shift(1).rolling(config.ROLLING_HOURS).mean()
    ).astype("float32")

    # Calendario
    when = table["hora_inicio"]
    table["hora"] = when.dt.hour.astype("int8")
    table["dia_semana"] = when.dt.day_of_week.astype("int8")
    table["mes"] = when.dt.month.astype("int8")
    table["anio"] = when.dt.year.astype("int16")
    table["es_fin_semana"] = (table["dia_semana"] >= 5).astype("int8")

    # Solo horas con servicio y con historial
    open_keys = _open_hours(demanda)
    row_key = (when.dt.year * 12 + when.dt.month).to_numpy(np.int64) * 24 + table["hora"].to_numpy(
        np.int64
    )
    is_open = np.isin(row_key, open_keys)
    has_history = table[LAG_COLUMNS].notna().all(axis=1).to_numpy()
    final = table[is_open & has_history].reset_index(drop=True)

    info = {
        "meses_cargados": len(d_files),
        "estaciones": n_stations,
        "filas_rejilla": total,
        "filas_sin_servicio_excluidas": int((~is_open).sum()),
        "filas_sin_historial_excluidas": int((is_open & ~has_history).sum()),
        "filas_finales": len(final),
        "proporcion_ceros": float((final[TARGET] == 0).mean()),
        "demanda_media": float(final[TARGET].mean()),
        "desde": desde or "",
        "hasta": hasta or "",
        "hora_min": str(final["hora_inicio"].min()),
        "hora_max": str(final["hora_inicio"].max()),
    }
    return final, info


def add_station_info(table: pd.DataFrame, catalog: pd.DataFrame | None) -> pd.DataFrame:
    """Agrega ubicación y capacidad de la estación, si el catálogo está disponible."""
    if catalog is None:
        return table
    info = catalog.rename(
        columns={
            "id": "estacion",
            "latitude": "latitud",
            "longitude": "longitud",
            "dpcapacity": "capacidad",
        }
    )
    info = info[["estacion", "latitud", "longitud", "capacidad"]].astype({"estacion": "int16"})
    return table.merge(info, on="estacion", how="left")


# ---------------------------------------------------------------------------------------------------------------
# Paso 3: preprocesamiento (normalización, codificación y reducción de dimensionalidad)
# ---------------------------------------------------------------------------------------------------------------
def split_temporal(table: pd.DataFrame, test_days: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrenamiento con el pasado y prueba con los últimos días: no se usa una división aleatoria."""
    cutoff = table["hora_inicio"].max().floor("D") - pd.Timedelta(days=test_days - 1)
    return table[table["hora_inicio"] < cutoff], table[table["hora_inicio"] >= cutoff]


def build_preprocessor():
    """Normaliza rezagos, codifica el calendario y la estación. Se ajusta solo con el entrenamiento (fit) y luego se aplica (transform)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    return ColumnTransformer(
        [
            ("normalizar", StandardScaler(), LAG_COLUMNS),
            ("calendario", OneHotEncoder(handle_unknown="ignore"), ["hora", "dia_semana", "mes"]),
            (
                "estacion",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["estacion"],
            ),
            ("fin_semana", "passthrough", ["es_fin_semana"]),
        ]
    )


def station_profile_pca(
    table: pd.DataFrame, n_components: int = 3
) -> tuple[pd.DataFrame, np.ndarray]:
    """Reduce el perfil semanal de cada estación (168 valores: día de la semana por hora) a pocos componentes."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    profile = (
        table.groupby(["estacion", "dia_semana", "hora"], observed=True)[TARGET]
        .mean()
        .unstack(["dia_semana", "hora"])
        .fillna(0.0)
    )
    scaled = StandardScaler().fit_transform(profile)
    pca = PCA(n_components=n_components, random_state=0).fit(scaled)
    components = pd.DataFrame(
        pca.transform(scaled),
        index=profile.index,
        columns=[f"pc{i + 1}" for i in range(n_components)],
    )
    return components.reset_index(), pca.explained_variance_ratio_


# ---------------------------------------------------------------------------------------------------------------
# Referencia: baseline estacional (la demanda de la misma hora de la semana anterior)
# ---------------------------------------------------------------------------------------------------------------
def baseline_metrics(table: pd.DataFrame, test_days: int = 7) -> dict:
    _, test = split_temporal(table, test_days)
    y = test[TARGET].to_numpy(np.float64)
    out = {"filas_prueba": float(len(test))}
    for name, column in (("estacional_168h", "lag_168"), ("ayer_24h", "lag_24")):
        error = y - test[column].to_numpy(np.float64)
        out[f"mae_{name}"] = float(np.abs(error).mean())
        out[f"rmse_{name}"] = float(np.sqrt((error**2).mean()))
        out[f"wape_{name}"] = float(np.abs(error).sum() / y.sum())
    return out

"""Pruebas de las reglas que más importan: limpieza, rejilla activa, rezagos sin fuga y lectura del índice de descargas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mibici_demand import dataset, features
from mibici_demand.cleaning import TripCleaner


def _trips(rows):
    df = pd.DataFrame(rows, columns=["Viaje_Id", "Usuario_Id", "Genero", "Año_de_nacimiento", "Inicio_del_viaje",
                                     "Fin_del_viaje", "Origen_Id", "Destino_Id"])
    df["Inicio_del_viaje"] = pd.to_datetime(df["Inicio_del_viaje"])
    df["Fin_del_viaje"] = pd.to_datetime(df["Fin_del_viaje"])
    return df


def test_limpieza_aplica_cada_regla_y_cuenta():
    base = ["2026-09-01 08:00:00", "2026-09-01 08:10:00", 1, 2]
    trips = _trips([
        [1, 10, "M", 1990, *base],                                           # viaje valido
        [1, 10, "M", 1990, *base],                                           # duplicado exacto
        [2, 11, "F", 199, "2026-09-01 09:00:00", "2026-09-01 09:10:00", 1, 2],  # anio de nacimiento imposible
        [3, 12, "M", 1985, "2026-09-01 10:00:00", "2026-09-01 10:00:05", 1, 1],  # anclaje fallido
        [4, 13, "M", 1985, "2026-09-01 11:00:00", "2026-09-03 11:00:00", 1, 2],  # mas de un dia
        [5, 14, "M", 1985, "2026-09-01 12:00:00", "2026-09-01 11:00:00", 1, 2],  # fin antes del inicio
        [6, 15, "M", 1985, pd.NaT, "2026-09-01 12:10:00", 1, 2],                # sin fecha de inicio
    ])
    clean, report = TripCleaner("limpio").clean(trips)
    assert report["sin_inicio_u_origen"] == 1
    assert report["duplicados_exactos"] == 1
    assert report["duracion_imposible"] == 2
    assert report["anclaje_fallido"] == 1
    assert report["anio_nacimiento_corregido"] == 1
    assert len(clean) == 2 and report["filas_finales"] == 2
    assert clean["Año_de_nacimiento"].isna().sum() == 1


def test_perfil_bruto_solo_exige_inicio_y_origen():
    trips = _trips([[1, 10, "M", 1990, "2026-09-01 08:00:00", "2026-09-01 08:00:05", 1, 1],
                    [1, 10, "M", 1990, "2026-09-01 08:00:00", "2026-09-01 08:00:05", 1, 1]])
    clean, _ = TripCleaner("bruto").clean(trips)
    assert len(clean) == 2


def _synthetic_month(tmp_path):
    """Septiembre de 2026, 21 días. La estación 1 existe todo el periodo; la 2 abre el día 10.
    Solo hay viajes de 6 a 22 h: las demás horas son horas sin servicio."""
    rng = np.random.default_rng(0)
    rows, trip_id = [], 0
    for day in range(1, 22):
        for hour in range(6, 23):
            for station in (1, 2):
                if station == 2 and day < 10:
                    continue
                for _ in range(int(rng.integers(0, 6))):
                    trip_id += 1
                    start = pd.Timestamp(2026, 9, day, hour, int(rng.integers(0, 50)))
                    destination = station if day < 10 else 3 - station  # antes del día 10 la estación 2 no recibe viajes
                    rows.append([trip_id, trip_id, "M", 1990, start, start + pd.Timedelta(minutes=12), station, destination])
    clean = _trips(rows)
    demanda, actividad = features.monthly_tables(clean)
    demanda.to_parquet(tmp_path / "demanda_2026-09.parquet", index=False)
    actividad.to_parquet(tmp_path / "actividad_2026-09.parquet", index=False)
    return clean


def test_rejilla_solo_donde_la_estacion_existia_y_con_servicio(tmp_path):
    _synthetic_month(tmp_path)
    table, info = features.build_station_hour_table(tmp_path)
    assert table["hora"].between(6, 22).all(), "las horas sin servicio no deben ser filas de entrenamiento"
    first_hour_station_2 = table.loc[table["estacion"] == 2, "hora_inicio"].min()
    assert first_hour_station_2 >= pd.Timestamp(2026, 9, 10), "no hay filas de la estación 2 antes de que abriera"
    assert info["filas_sin_servicio_excluidas"] > 0 and info["filas_sin_historial_excluidas"] > 0


def test_rezagos_usan_solo_informacion_anterior(tmp_path):
    clean = _synthetic_month(tmp_path)
    table, _ = features.build_station_hour_table(tmp_path)
    counts = clean.assign(h=clean["Inicio_del_viaje"].dt.floor("h")).groupby(["Origen_Id", "h"]).size()

    def demand(station, ts):
        return float(counts.get((station, ts), 0))

    row = table[(table["estacion"] == 1) & (table["hora_inicio"] == pd.Timestamp(2026, 9, 15, 12))].iloc[0]
    ts = pd.Timestamp(2026, 9, 15, 12)
    assert row["lag_1"] == demand(1, ts - pd.Timedelta(hours=1))
    assert row["lag_24"] == demand(1, ts - pd.Timedelta(hours=24))
    assert row["lag_168"] == demand(1, ts - pd.Timedelta(hours=168))
    previous_24 = np.mean([demand(1, ts - pd.Timedelta(hours=k)) for k in range(1, 25)])
    assert np.isclose(row["media_24h"], previous_24), "la media móvil no incluye la hora que se predice"
    assert row["demanda"] == demand(1, ts)


def test_division_temporal_no_mezcla_futuro_en_entrenamiento(tmp_path):
    _synthetic_month(tmp_path)
    table, _ = features.build_station_hour_table(tmp_path)
    train, test = features.split_temporal(table, test_days=3)
    assert train["hora_inicio"].max() < test["hora_inicio"].min()


def test_indice_de_descargas_toma_un_archivo_por_mes():
    page = """
    [Diciembre](https://mibici.net/site/assets/files/1079/datos_abiertos_2014_12.csv)
    [January](https://mibici.net/site/assets/files/1020/datos_abiertos_2016_01-1.csv)
    <a href="https://mibici.net/site/assets/files/1021/datos_abiertos_2016_02.csv">February</a>
    [Nomenclatura](https://mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv)
    """
    files, catalog = dataset.parse_index(page)
    assert [f.period for f in files] == ["2014-12", "2016-01", "2016-02"]
    assert files[1].filename == "datos_abiertos_2016_01.csv"
    assert catalog.endswith("nomenclatura_2026_09.csv")


def test_indice_acepta_enlaces_relativos():
    page = '<a href="/site/assets/files/1079/datos_abiertos_2014_12.csv">Diciembre</a> <a href="//mibici.net/site/assets/files/1021/datos_abiertos_2016_02.csv">Febrero</a>'
    files, catalog = dataset.parse_index(page)
    assert [f.url for f in files] == [
        "https://mibici.net/site/assets/files/1079/datos_abiertos_2014_12.csv",
        "https://mibici.net/site/assets/files/1021/datos_abiertos_2016_02.csv",
    ]
    assert catalog is None


def test_lectura_tolera_cambios_de_encabezado(tmp_path):
    header = '"Viaje_Id","Usuario_Id","Genero","{nac}","Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"\n'
    row = '"1","2","M","1990","2026-09-01 08:00:00","2026-09-01 08:10:00","3","4"\n'
    for name, nac in (("a", "Año_de_nacimiento"), ("b", "Ano_de_nacimiento"), ("c", "Año de nacimiento")):
        path = tmp_path / f"datos_abiertos_2026_09_{name}.csv"
        path.write_text(header.format(nac=nac) + row, encoding="utf-8")
        trips = dataset.read_trips(path)
        assert trips.loc[0, "Año_de_nacimiento"] == 1990 and trips.attrs["columnas_faltantes"] == []
    sin_nacimiento = tmp_path / "datos_abiertos_2026_09_d.csv"
    sin_nacimiento.write_text('"Viaje_Id","Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"\n"1","2026-09-01 08:00:00","2026-09-01 08:10:00","3","4"\n', encoding="utf-8")
    trips = dataset.read_trips(sin_nacimiento)
    assert trips["Año_de_nacimiento"].isna().all() and "Año_de_nacimiento" in trips.attrs["columnas_faltantes"]


def _csv_mes(tmp_path, nombre, fechas):
    header = '"Viaje_Id","Usuario_Id","Genero","Año_de_nacimiento","Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"\n'
    rows = "".join(f'"{i}","2","M","1990","{f} 08:00:00","{f} 08:10:00","3","4"\n' for i, f in enumerate(fechas, 1))
    path = tmp_path / nombre
    path.write_text(header + rows, encoding="utf-8")
    return path


def test_lectura_detecta_fechas_con_dia_y_mes_invertidos(tmp_path):
    # Noviembre de 2023 escrito como AAAA-DD-MM: el 13 de noviembre aparece como "2023-13-11"
    invertido = _csv_mes(tmp_path, "datos_abiertos_2023_11.csv", [f"2023-{d:02d}-11" for d in range(1, 31)])
    trips = dataset.read_trips(invertido)
    assert trips.attrs["formato_fecha"] == "%Y-%d-%m %H:%M:%S"
    assert trips["Inicio_del_viaje"].notna().all() and (trips["Inicio_del_viaje"].dt.month == 11).all()
    assert trips["Inicio_del_viaje"].dt.day.nunique() == 30


def test_lectura_conserva_el_formato_normal(tmp_path):
    normal = _csv_mes(tmp_path, "datos_abiertos_2023_10.csv", [f"2023-10-{d:02d}" for d in range(1, 31)])
    trips = dataset.read_trips(normal)
    assert trips.attrs["formato_fecha"] == "AAAA-MM-DD" and (trips["Inicio_del_viaje"].dt.month == 10).all()


def test_lectura_reconoce_encabezados_con_codificacion_danada(tmp_path):
    for nombre, nac in (("a", "A\ufffdo_de_nacimiento"), ("b", "A}\ufffde_nacimiento")):
        header = f'"Viaje_Id","Usuario_Id","Genero","{nac}","Inicio_del_viaje","Fin_del_viaje","Origen_Id","Destino_Id"\n'
        path = tmp_path / f"datos_abiertos_2024_02_{nombre}.csv"
        path.write_text(header + '"1","2","M","1990","2024-02-01 08:00:00","2024-02-01 08:10:00","3","4"\n', encoding="utf-8")
        assert dataset.read_trips(path).loc[0, "Año_de_nacimiento"] == 1990


def test_lectura_detecta_formato_dia_mes_anio_sin_segundos(tmp_path):
    # Noviembre de 2023 se publicó como DD/MM/AAAA h:mm; con días del 13 en adelante la lectura automática los pierde
    ruta = _csv_mes(tmp_path, "datos_abiertos_2023_11.csv", [f"{d:02d}/11/2023" for d in range(1, 31)])
    texto = ruta.read_text(encoding="utf-8").replace(" 08:00:00", " 8:00").replace(" 08:10:00", " 8:10")
    ruta.write_text(texto, encoding="utf-8")
    trips = dataset.read_trips(ruta)
    assert trips.attrs["formato_fecha"] == "%d/%m/%Y %H:%M"
    assert trips["Inicio_del_viaje"].notna().all() and (trips["Inicio_del_viaje"].dt.month == 11).all()


def test_formato_dia_mes_se_detecta_aunque_solo_haya_dias_del_1_al_12(tmp_path):
    # Con días <= 12 la lectura automática no falla, pero deja los viajes en otro mes: se elige el formato que los deja en su mes
    ruta = _csv_mes(tmp_path, "datos_abiertos_2023_11.csv", [f"{d:02d}/11/2023" for d in range(1, 13)])
    ruta.write_text(ruta.read_text(encoding="utf-8").replace(" 08:00:00", " 8:00").replace(" 08:10:00", " 8:10"), encoding="utf-8")
    trips = dataset.read_trips(ruta)
    assert (trips["Inicio_del_viaje"].dt.month == 11).all() and trips["Inicio_del_viaje"].dt.day.nunique() == 12


def test_las_figuras_se_generan(tmp_path):
    from mibici_demand import plots

    meses = pd.period_range("2019-01", "2021-12", freq="M").astype(str)
    resumen = pd.DataFrame({
        "mes": meses, "filas_originales": 1000, "filas_finales": 950, "estaciones_activas": 100,
        "sin_inicio_u_origen": 0, "duplicados_exactos": 1, "duracion_imposible": 2, "anclaje_fallido": 47, "anio_nacimiento_corregido": 3,
    })
    rng = np.random.default_rng(1)
    tabla = pd.DataFrame({
        "estacion": np.repeat(np.arange(1, 31), 24 * 7).astype("int16"),
        "dia_semana": np.tile(np.repeat(np.arange(7), 24), 30), "hora": np.tile(np.arange(24), 30 * 7),
        "demanda": rng.poisson(2, 30 * 24 * 7),
    })
    tabla = tabla[tabla["hora"].between(5, 23)]
    components, explained = features.station_profile_pca(tabla)
    rutas = [plots.trips_per_month(resumen, tmp_path), plots.cleaning_by_rule(resumen, tmp_path), plots.demand_heatmap(tabla, tmp_path),
             plots.demand_distribution(tabla, tmp_path), plots.pca_stations(components, explained, tmp_path, tabla)]
    assert all(ruta.exists() and ruta.stat().st_size > 0 for ruta in rutas)

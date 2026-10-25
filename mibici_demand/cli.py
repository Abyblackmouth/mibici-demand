"""Línea de comandos del pipeline.

python -m mibici_demand.cli descargar [--desde 2014-12 --hasta 2026-09]
python -m mibici_demand.cli procesar  [--perfil limpio|bruto] [--desde ... --hasta ...]
python -m mibici_demand.cli construir [--perfil limpio|bruto] [--desde ... --hasta ...] [--etiqueta desde2022]
python -m mibici_demand.cli registrar [--perfil limpio|bruto] [--version v1] [--etiqueta desde2022]
python -m mibici_demand.cli figuras   [--perfil limpio]
python -m mibici_demand.cli todo      [--perfil limpio] [--desde ... --hasta ...]
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd

from mibici_demand import config, dataset, features, plots, tracking
from mibici_demand.cleaning import TripCleaner


def _interim(perfil: str) -> Path:
    return config.INTERIM_DIR / perfil


def _suffix(etiqueta: str | None) -> str:
    return f"_{etiqueta}" if etiqueta else ""


def _processed_path(perfil: str, etiqueta: str | None = None) -> Path:
    return config.PROCESSED_DIR / f"demanda_estacion_hora_{perfil}{_suffix(etiqueta)}.parquet"


def _summary_path(perfil: str, etiqueta: str | None = None) -> Path:
    return config.PROCESSED_DIR / f"resumen_tabla_{perfil}{_suffix(etiqueta)}.json"


def cmd_descargar(args) -> None:
    manifest = dataset.download(args.desde, args.hasta)
    print(f"{len(manifest)} archivos en data/raw; manifiesto en data/raw/manifest.csv")


def cmd_procesar(args) -> None:
    """Lee cada mes por separado, lo limpia y guarda sus tablas pequeñas: nunca hay más de un mes en memoria."""
    files = dataset.list_raw_files(args.desde, args.hasta)
    if not files:
        raise SystemExit("no hay archivos en data/raw para esa ventana; corre primero 'descargar'")
    cleaner = TripCleaner(args.perfil)
    out = _interim(args.perfil)
    out.mkdir(parents=True, exist_ok=True)
    rows, failed = [], []
    for path in files:
        period = dataset.period_of(path)
        try:
            trips = dataset.read_trips(path)
            clean, report = cleaner.clean(trips)
            if report["filas_finales"] == 0:
                raise ValueError(
                    f"{path.name}: ningún viaje con fecha y origen válidos; revisa el formato de las fechas"
                )
            demanda, actividad = features.monthly_tables(clean)
        except (
            ValueError,
            KeyError,
        ) as error:  # un mes con problemas no debe tirar el resto, pero se reporta y el comando falla al final
            failed.append((period, str(error)))
            print(f"{period}: ERROR -> {error}")
            continue
        demanda.to_parquet(out / f"demanda_{period}.parquet", index=False)
        actividad.to_parquet(out / f"actividad_{period}.parquet", index=False)
        report.update(
            {
                "mes": period,
                "estaciones_activas": int(actividad["estacion"].nunique()),
                "columnas_faltantes": ",".join(trips.attrs.get("columnas_faltantes", [])),
                "formato_fecha": trips.attrs.get("formato_fecha", ""),
                "fuera_del_mes": int(
                    (clean["Inicio_del_viaje"].dt.strftime("%Y-%m") != period).sum()
                ),
            }
        )
        rows.append(report)
        print(f"{period}: {report['filas_originales']:,} -> {report['filas_finales']:,} viajes")
        if (
            report["formato_fecha"] != "AAAA-MM-DD"
            or report["fuera_del_mes"] > 0.05 * report["filas_finales"]
            or report["sin_inicio_u_origen"] > 0.01 * report["filas_originales"]
        ):
            print(
                f"   AVISO {period}: formato de fecha {report['formato_fecha']}; sin fecha u origen: "
                f"{report['sin_inicio_u_origen']:,}; fuera de su mes: {report['fuera_del_mes']:,}"
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "resumen_limpieza.csv", index=False)
    if failed:
        print(f"\nMeses con error ({len(failed)}): {[p for p, _ in failed]}")
        raise SystemExit(1)


def cmd_construir(args) -> None:
    table, info = features.build_station_hour_table(_interim(args.perfil), args.desde, args.hasta)
    table = features.add_station_info(table, dataset.read_catalog())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(_processed_path(args.perfil, args.etiqueta), index=False)
    _summary_path(args.perfil, args.etiqueta).write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    print(json.dumps(info, indent=2))


def _table_sample(path: Path, rows: int = 10_000) -> pd.DataFrame:
    """Primeras filas de la tabla con todas sus columnas: representan su esquema en MLflow sin cargar los millones de filas."""
    import pyarrow.parquet as pq

    return next(pq.ParquetFile(path).iter_batches(batch_size=rows)).to_pandas()


def cmd_registrar(args) -> None:
    """Registra en MLflow las tres versiones de datos: archivos crudos, viajes limpios y tabla de entrenamiento."""
    out = _interim(args.perfil)
    raw_manifest = config.RAW_DIR / "manifest.csv"
    if not raw_manifest.exists():
        dataset.write_manifest()
    manifest = pd.read_csv(raw_manifest)
    summary = pd.read_csv(out / "resumen_limpieza.csv")
    table_path = _processed_path(args.perfil, args.etiqueta)
    # Solo las columnas que usan las métricas: la tabla completa tiene millones de filas y no cabría junto con sus copias
    table = pd.read_parquet(
        table_path,
        columns=["estacion", "hora_inicio", "demanda", "hora", "dia_semana", "lag_24", "lag_168"],
    )
    info = json.loads(_summary_path(args.perfil, args.etiqueta).read_text(encoding="utf-8"))
    window = {
        "desde": info["desde"] or "inicio",
        "hasta": info["hasta"] or "fin",
        "primera_hora": info["hora_min"],
        "ultima_hora": info["hora_max"],
        "perfil": args.perfil,
    }
    base_tags = {"stage": "fase-1", "perfil": args.perfil, "ventana": args.etiqueta or "completa"}

    if not args.etiqueta:  # los archivos crudos y la limpieza son los mismos para todas las ventanas: se registran una vez
        tracking.log_dataset_version(
            "archivos_crudos",
            args.version,
            manifest,
            config.RAW_DIR,
            context="raw",
            params={"fuente": config.OPEN_DATA_URL, "perfil": args.perfil},
            metrics={"archivos": len(manifest), "bytes_totales": manifest["bytes"].sum()},
            artifacts=[raw_manifest],
            tags=base_tags,
        )

        cleaning_metrics = {
            f"total_{c}": summary[c].sum()
            for c in [
                "filas_originales",
                "sin_inicio_u_origen",
                "duplicados_exactos",
                "duracion_imposible",
                "anclaje_fallido",
                "anio_nacimiento_corregido",
                "filas_finales",
            ]
        }
        tracking.log_dataset_version(
            "viajes_limpios",
            args.version,
            summary,
            out,
            context="cleaning",
            params={
                "min_segundos": config.MIN_TRIP_SECONDS,
                "max_segundos": config.MAX_TRIP_SECONDS,
                "primer_mes": summary["mes"].min(),
                "ultimo_mes": summary["mes"].max(),
                "perfil": args.perfil,
            },
            metrics=cleaning_metrics,
            artifacts=[out / "resumen_limpieza.csv"],
            tags=base_tags,
        )

    _, explained = features.station_profile_pca(table)
    metrics = {
        "filas": len(table),
        "estaciones": table["estacion"].nunique(),
        "proporcion_ceros": float((table["demanda"] == 0).mean()),
        "varianza_pca_3_componentes": float(explained.sum()),
        **features.baseline_metrics(table),
    }
    rows_in_table = len(table)
    del table
    gc.collect()
    tracking.log_dataset_version(
        "demanda_estacion_hora",
        args.version,
        _table_sample(table_path),
        table_path,
        context="training",
        params={
            "rezagos": list(config.LAGS),
            "horas_cerradas_share": config.CLOSED_HOUR_SHARE,
            "filas_tabla_completa": rows_in_table,
            "sha256_tabla": dataset.sha256_file(table_path),
            "registro_en_mlflow": "muestra de las primeras 10,000 filas; la tabla completa se identifica con sha256_tabla",
            **window,
        },
        metrics=metrics,
        artifacts=[_summary_path(args.perfil, args.etiqueta)],
        tags=base_tags,
    )
    print(
        f"versiones registradas en MLflow ({config.TRACKING_URI}), experimento '{config.EXPERIMENT_NAME}'"
    )


def cmd_figuras(args) -> None:
    summary = pd.read_csv(_interim(args.perfil) / "resumen_limpieza.csv")
    table = pd.read_parquet(_processed_path(args.perfil, args.etiqueta))
    components, explained = features.station_profile_pca(table)
    out_dir = config.FIGURES_DIR / args.etiqueta if args.etiqueta else config.FIGURES_DIR
    for path in (
        plots.trips_per_month(summary, out_dir),
        plots.cleaning_by_rule(summary, out_dir),
        plots.demand_heatmap(table, out_dir),
        plots.demand_distribution(table, out_dir),
        plots.pca_stations(components, explained, out_dir, table),
    ):
        print("figura:", path)


def cmd_todo(args) -> None:
    cmd_procesar(args)
    cmd_construir(args)
    cmd_figuras(args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="mibici_demand",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, func in (
        ("descargar", cmd_descargar),
        ("procesar", cmd_procesar),
        ("construir", cmd_construir),
        ("registrar", cmd_registrar),
        ("figuras", cmd_figuras),
        ("todo", cmd_todo),
    ):
        p = sub.add_parser(name)
        p.add_argument("--desde", help="primer mes, formato AAAA-MM")
        p.add_argument("--hasta", help="último mes, formato AAAA-MM")
        p.add_argument("--perfil", default="limpio", choices=["limpio", "bruto"])
        p.add_argument("--version", default="v1", help="etiqueta de la versión de datos en MLflow")
        p.add_argument(
            "--etiqueta",
            default="",
            help="nombre de la ventana (por ejemplo desde2022) para guardar otra versión de la tabla",
        )
        p.set_defaults(func=func)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

"""Prepara snapshot causal y candidato sin entrenar ni sobrescribir evidencia."""

from __future__ import annotations

import json
import shutil
import subprocess

import pandas as pd
from pandas.testing import assert_frame_equal

from mibici_demand import config, dataset
from mibici_demand.modeling.backtest import MODEL, write_json
from mibici_demand.temporal import BacktestSpec, build_fold


def main():
    root = config.ROOT
    if subprocess.check_output(["git", "branch", "--show-current"], text=True).strip() != "fase3":
        raise RuntimeError("Solo ejecutar en fase3")
    reports = root / "reports/fase3"
    execution = json.loads((reports / "ejecucion.json").read_text(encoding="utf-8"))
    audits = json.loads((reports / "cortes.json").read_text(encoding="utf-8"))
    trips_path = root / "data/processed/fase3/viajes_limpios.parquet"
    predictions_path = root / "data/processed/fase3/predicciones_backtest.parquet"
    assert dataset.sha256_file(trips_path) == execution["procedencia"]["viajes_limpios_sha256"]
    assert dataset.sha256_file(predictions_path) == execution["predicciones_sha256"]
    trips, predictions = pd.read_parquet(trips_path), pd.read_parquet(predictions_path)
    spec = BacktestSpec(**execution["configuracion"])
    destination = root / "data/snapshots/fase3-temporal-v1"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": "fase3-temporal-v1",
        "origen_ejecucion": execution["id"],
        "procedencia": execution["procedencia"],
        "configuracion": execution["configuracion"],
        "modelo": MODEL,
        "parametros": execution["parametros_modelo"],
        "semilla": config.RANDOM_SEED,
        "threads": 4,
        "tolerancias_predeclaradas": {"metricas_atol": 1e-6, "metricas_rtol": 1e-6},
        "cortes": [],
    }
    final_train = None
    for cutoff, audit in zip(spec.timestamps, audits):
        train, evaluation, rebuilt = build_fold(trips, cutoff, spec)
        assert rebuilt["train_fin_exclusivo"] == audit["train_fin_exclusivo"]
        saved = predictions[predictions["corte"].eq(str(cutoff))].reset_index(drop=True)
        assert_frame_equal(evaluation, saved[evaluation.columns], check_exact=True)
        evaluation["prediccion_referencia"] = saved[MODEL]
        files = []
        for name, frame in (("train", train), ("evaluation", evaluation)):
            path = destination / f"{cutoff:%Y%m%d}_{name}.parquet"
            if path.exists():
                assert_frame_equal(frame, pd.read_parquet(path), check_exact=True)
            else:
                frame.to_parquet(path, index=False)
            files.append(
                {
                    "tipo": name,
                    "ruta": path.relative_to(root).as_posix(),
                    "sha256": dataset.sha256_file(path),
                    "bytes": path.stat().st_size,
                    "filas": len(frame),
                    "dtypes": frame.dtypes.astype(str).to_dict(),
                }
            )
        manifest["cortes"].append(
            {
                "corte": str(cutoff),
                "archivos": files,
                "columnas": audit["columnas_entrada"],
                "auditoria": audit,
            }
        )
        final_train = train
        print("Snapshot verificado:", cutoff, len(train), len(evaluation), flush=True)
    write_json(root / "references/data_snapshot.json", manifest)
    # Política cronológica; no ordenar por métricas de evaluación conocidas.
    chosen = max(execution["cortes"], key=lambda c: pd.Timestamp(c["corte"]))
    audit = next(a for a in audits if a["corte"] == chosen["corte"])
    original = root / chosen["modelo_local"]
    final = root / "models/final/pipeline.joblib"
    final.parent.mkdir(parents=True, exist_ok=True)
    assert dataset.sha256_file(original) == audit["modelo_sha256"]
    if final.exists():
        assert dataset.sha256_file(final) == audit["modelo_sha256"], "No sobrescribir otro release"
    else:
        shutil.copy2(original, final)
    release = {
        "release_id": "mibici-temporal-20260921-v1",
        "model_path": "models/final/pipeline.joblib",
        "sha256": dataset.sha256_file(final),
        "source_model": chosen["modelo_local"],
        "source_mlflow_run_id": chosen["run_id"],
        "source_mlflow_model_uri": chosen["mlflow_model_uri"],
        "corte": chosen["corte"],
        "train_inicio": audit["train_inicio"],
        "train_fin_exclusivo": audit["train_fin_exclusivo"],
        "uso": "candidato_temporal_integracion_no_aprobado_produccion",
        "produccion_aprobada": False,
        "seleccion": "Ultimo corte causal por cronologia; no argmin de metricas de evaluacion",
        "criterios": {
            "trazabilidad_causal": True,
            "sin_octubre": True,
            "robustez_familia_acreditada": False,
            "motivo_no_aprobacion": "Primer corte: extremo 438.0117 y RMSE peor que baselines; no hay holdout nuevo certificado",
        },
        "input_dtypes": {c: str(final_train[c].dtype) for c in audit["columnas_entrada"]},
        "snapshot_manifest": "references/data_snapshot.json",
        "snapshot_manifest_sha256": dataset.sha256_file(root / "references/data_snapshot.json"),
        "historico_sha256": dataset.sha256_file(root / "models/best_pipeline.joblib"),
    }
    write_json(root / "references/model_release.json", release)
    plan = json.loads((root / "references/plan_drift.json").read_text(encoding="utf-8"))
    plan.update(
        corte="2026-09-21",
        seleccion=release["seleccion"],
        referencia="Entrenamiento del ultimo corte temporal; solo distribuciones",
        monitoreo="2026-09-21 a 2026-09-28 exclusivo, mismo artefacto de API y Docker",
        estado="Extiende sin modificar los umbrales de la simulacion previa; resultados previos ya conocidos",
    )
    write_json(root / "references/plan_drift_final.json", plan)
    # Ejemplo posterior al entrenamiento; rezagos reales del parquet auditado.
    last = predictions[predictions["corte"].eq(chosen["corte"])]
    rows = []
    for station, hour in ((2, 8), (46, 21)):
        row = last[
            last["estacion"].eq(station) & last["hora_inicio"].eq(pd.Timestamp(2026, 9, 26, hour))
        ].iloc[0]
        rows.append(
            {
                "estacion": station,
                "hora_inicio": f"2026-09-26T{hour:02d}:00:00-06:00",
                "lag_24": float(row["lag_24"]),
                "lag_168": float(row["lag_168"]),
            }
        )
    rows.append({**rows[0], "estacion": 999})
    write_json(
        root / "examples/predict_request_final.json",
        {"origen_pronostico": "2026-09-26T00:00:00-06:00", "observaciones": rows},
    )
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

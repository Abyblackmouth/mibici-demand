"""Reajusta los tres pipelines desde snapshots inmutables, sin promover modelos.

No busca hiperparámetros. Compara con las predicciones guardadas, incluyendo
extremos. Usa exactamente build_model_pipeline y los parámetros de fit_fold.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from mibici_demand import config
from mibici_demand.modeling.pipeline import build_model_pipeline


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def metrics(y, prediction) -> dict:
    y, prediction = np.asarray(y, dtype="float64"), np.asarray(prediction, dtype="float64")
    if y.shape != prediction.shape or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Predicciones inválidas; no se omiten ni recortan")
    error = y - prediction
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.abs(error).sum() / y.sum()) if y.sum() else None,
        "r2": float(1 - (error**2).sum() / ((y - y.mean()) ** 2).sum()) if np.var(y) > 0 else None,
    }


def reproduce(output: Path) -> dict:
    manifest_path = config.ROOT / "references/data_snapshot.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = output.resolve()
    if (
        output.is_relative_to(config.MODELS_DIR.resolve())
        or output == config.REPORTS_DIR.resolve()
    ):
        raise ValueError("No escribir sobre modelos seleccionados o reportes históricos")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for cut in manifest["cortes"]:
        frames = {}
        for item in cut["archivos"]:
            path = config.ROOT / item["ruta"]
            if digest(path) != item["sha256"]:
                raise ValueError(f"Snapshot adulterado: {path}")
            frames[item["tipo"]] = pd.read_parquet(path)
            if frames[item["tipo"]].dtypes.astype(str).to_dict() != item["dtypes"]:
                raise ValueError("Tipos distintos de la versión del snapshot")
        train, evaluation = frames["train"], frames["evaluation"]
        cutoff = pd.Timestamp(cut["corte"])
        assert train.hora_inicio.max() < cutoff <= evaluation.hora_inicio.min()
        for frame in (train, evaluation):
            for lag in (24, 168):
                assert (
                    frame[f"fuente_lag_{lag}"] + pd.Timedelta(hours=1) <= frame.origen_pronostico
                ).all()
        model = build_model_pipeline(
            manifest["modelo"],
            params=manifest["parametros"],
            horizon="24h",
            include_station_meta=False,
        )
        with threadpool_limits(limits=manifest["threads"]):
            model.fit(train[cut["columnas"]], train["demanda"])
            predicted = model.predict(evaluation[cut["columnas"]])
        reference = evaluation["prediccion_referencia"].to_numpy()
        actual, expected = (
            metrics(evaluation.demanda, predicted),
            metrics(evaluation.demanda, reference),
        )
        tolerance = manifest["tolerancias_predeclaradas"]
        passed = all(
            np.isclose(
                actual[k],
                expected[k],
                atol=tolerance["metricas_atol"],
                rtol=tolerance["metricas_rtol"],
            )
            for k in actual
        )
        artifact = output / f"pipeline_{cutoff:%Y%m%d}.joblib"
        joblib.dump(model, artifact)
        results.append(
            {
                "corte": cut["corte"],
                "filas_train": len(train),
                "filas_evaluacion": len(evaluation),
                "metricas": actual,
                "referencia": expected,
                "metricas_en_tolerancia": bool(passed),
                "diferencia_maxima_predicciones": float(np.abs(predicted - reference).max()),
                "diferencias_metricas": {k: actual[k] - expected[k] for k in actual},
                "modelo_reentrenado_sha256": digest(artifact),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    result = {
        "estado": "reproducido"
        if all(r["metricas_en_tolerancia"] for r in results)
        else "diferencias_fuera_tolerancia",
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "executable": sys.executable,
        "snapshot_sha256": digest(manifest_path),
        "cortes": results,
        "tolerancias": manifest["tolerancias_predeclaradas"],
        "promocion_ejecutada": False,
        "versiones": {
            p: importlib.metadata.version(p)
            for p in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "joblib",
                "pyarrow",
                "threadpoolctl",
            )
        },
    }
    (output / "resultado.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce(args.output)
    if result["estado"] != "reproducido":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

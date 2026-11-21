"""Registra procedencia y candidato en una base nueva; no cambia champion histórico."""

import json
from pathlib import Path

import joblib
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from threadpoolctl import threadpool_limits

from mibici_demand import config, dataset
from mibici_demand.modeling.backtest import write_json
from mibici_demand.release import read_release


def main():
    release = read_release()
    root = config.ROOT
    model_path = root / release["model_path"]
    assert dataset.sha256_file(model_path) == release["sha256"]
    out = root / "reports/fase3/integracion"
    out.mkdir(parents=True, exist_ok=True)
    uri = f"sqlite:///{(root / 'mlflow_integracion.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    name = "mibici-fase3-integracion-final"
    experiment = client.get_experiment_by_name(name)
    experiment_id = (
        experiment.experiment_id
        if experiment
        else client.create_experiment(
            name, artifact_location=(root / "mlartifacts/integracion").as_uri()
        )
    )
    model = joblib.load(model_path)
    manifest = json.loads((root / release["snapshot_manifest"]).read_text(encoding="utf-8"))
    last = manifest["cortes"][-1]
    evaluation = pd.read_parquet(root / last["archivos"][1]["ruta"])
    sample = evaluation[last["columnas"]].head(5)
    with threadpool_limits(limits=4):
        expected = model.predict(sample)
    # OneDrive conserva atributos de solo lectura en carpetas originales.
    # Copiar bytes a un staging nuevo evita que MLflow herede esos atributos al
    # limpiar su TemporalDirectory; no se modifica el árbol fuente ni se oculta el fallo.
    staged = root / ".venv/mlflow-code-staging/mibici_demand"
    for name in (
        "__init__.py",
        "config.py",
        "features.py",
        "modeling/__init__.py",
        "modeling/pipeline.py",
    ):
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / "mibici_demand" / name).read_bytes())
    with mlflow.start_run(experiment_id=experiment_id, run_name=release["release_id"]) as run:
        mlflow.log_params(
            {
                "source_run_id": release["source_mlflow_run_id"],
                "source_model_uri": release["source_mlflow_model_uri"],
                "joblib_sha256": release["sha256"],
                "snapshot_sha256": release["snapshot_manifest_sha256"],
                "corte": release["corte"],
                "train_inicio": release["train_inicio"],
                "train_fin_exclusivo": release["train_fin_exclusivo"],
                "semilla": manifest["semilla"],
                "threads": manifest["threads"],
                **manifest["parametros"],
            }
        )
        mlflow.set_tags(
            {
                "stage": "fase3-integracion",
                "uso": release["uso"],
                "production_approved": "false",
                "seleccion": "cronologia_no_ranking_test",
            }
        )
        mlflow.log_artifact(str(model_path), "artefacto_exacto")
        mlflow.log_artifact(str(root / "references/model_release.json"), "procedencia")
        mlflow.log_artifact(str(root / "references/data_snapshot.json"), "procedencia")
        logged = mlflow.sklearn.log_model(
            model,
            name="pipeline",
            input_example=sample,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            signature=infer_signature(sample, expected),
            registered_model_name="mibici-fase3-candidato",
            code_paths=[str(staged)],
            pip_requirements=str(root / "requirements-linux-py313.txt"),
        )
        version = client.search_model_versions("name='mibici-fase3-candidato'")
        current = next(v for v in version if v.run_id == run.info.run_id)
        client.set_registered_model_alias("mibici-fase3-candidato", "candidate", current.version)
        downloaded = Path(
            client.download_artifacts(
                run.info.run_id, "artefacto_exacto/pipeline.joblib", str(root / ".venv")
            )
        )
        assert dataset.sha256_file(downloaded) == release["sha256"]
        write_json(
            out / "mlflow_release.json",
            {
                "tracking_uri": uri,
                "run_id": run.info.run_id,
                "registered_model": "mibici-fase3-candidato",
                "version": current.version,
                "alias": "candidate",
                "model_uri": logged.model_uri,
                "exact_joblib_uri": f"runs:/{run.info.run_id}/artefacto_exacto/pipeline.joblib",
                "joblib_sha256_recuperado": dataset.sha256_file(downloaded),
                "historical_champion_modified": False,
            },
        )


if __name__ == "__main__":
    main()

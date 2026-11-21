"""Integra candidato: Python Windows, HTTP Windows y Linux; preserva evidencias previas."""

from hashlib import sha256
import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import requests
from scripts.verify_docker import container_service, docker, values, windows_service
from threadpoolctl import threadpool_limits

from mibici_demand import api, config
from mibici_demand.modeling.backtest import write_json
from mibici_demand.release import default_model_path, read_release

IMAGE = "mibici-demand:fase3"


def run():
    release = read_release()
    out = config.ROOT / "reports/fase3/integracion"
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(
        (config.ROOT / "examples/predict_request_final.json").read_text(encoding="utf-8")
    )
    # Constructor independiente de la API; exactamente los dtypes del snapshot.
    import pandas as pd

    records = []
    for row in payload["observaciones"]:
        ts = pd.Timestamp(row["hora_inicio"]).tz_convert("America/Mexico_City")
        records.append(
            {
                "estacion": row["estacion"],
                "hora": ts.hour,
                "dia_semana": ts.day_of_week,
                "mes": ts.month,
                "es_fin_semana": int(ts.day_of_week >= 5),
                "lag_24": row["lag_24"],
                "lag_168": row["lag_168"],
            }
        )
    frame = pd.DataFrame(records).astype(release["input_dtypes"])
    with threadpool_limits(limits=4):
        direct = joblib.load(default_model_path()).predict(frame)
    with TestClient(api.create_app(default_model_path())) as connection:
        np.testing.assert_allclose(
            values(connection.post("/predict", json=payload).json()),
            direct,
            rtol=1e-12,
            atol=1e-12,
        )
    with (
        container_service(IMAGE) as linux,
        windows_service(default_model_path()) as (windows_url, _),
    ):
        responses = []
        status = []
        for base in (windows_url, linux.url):
            health = requests.get(base + "/health", timeout=5)
            docs = requests.get(base + "/docs", timeout=5)
            response = requests.post(base + "/predict", json=payload, timeout=10)
            invalid = json.loads(json.dumps(payload))
            invalid["observaciones"][0]["lag_24"] = None
            bad = requests.post(base + "/predict", json=invalid, timeout=5)
            assert (
                health.status_code == docs.status_code == response.status_code == 200
                and bad.status_code == 422
            )
            assert health.json()["model"]["artifact_sha256"] == release["sha256"]
            assert health.json()["model"]["release_id"] == release["release_id"]
            np.testing.assert_allclose(values(response.json()), direct, rtol=1e-12, atol=1e-12)
            responses.append(response.json())
            status.append(
                {
                    "health": health.status_code,
                    "docs": docs.status_code,
                    "predict": response.status_code,
                    "invalid": bad.status_code,
                }
            )
        source = r"""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from mibici_demand.api import load_model
p=Path(os.environ['MODEL_PATH']);model=load_model(p)
data=json.loads(input());frame=pd.DataFrame(data['records']).astype(data['dtypes'])
with threadpool_limits(limits=4): values=model.pipeline.predict(frame).tolist()
print(json.dumps({'predictions':values,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'model_path':str(p),'dtypes':frame.dtypes.astype(str).to_dict(),'prefix':sys.prefix,'base_prefix':sys.base_prefix,'uid':os.getuid(),'mlflow_installed':importlib.util.find_spec('mlflow') is not None,'forbidden':[str(p) for p in [Path('/app/.venv'),Path('/app/data'),Path('/app/.env'),Path('/app/mlflow.db')] if p.exists()],'files':{p.relative_to('/app').as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/app').rglob('*') if p.is_file()}}))
"""
        # source via exec para reservar stdin al JSON; no shell ni sustitución de comandos.
        native = json.loads(
            docker(
                "exec",
                "-i",
                linux.identifier,
                "python",
                "-c",
                source,
                stdin=json.dumps({"records": records, "dtypes": release["input_dtypes"]}),
            )
        )
        np.testing.assert_allclose(native["predictions"], direct, rtol=1e-12, atol=1e-12)
        assert native["sha256"] == release["sha256"]
        assert linux.info["Mounts"] == [] and linux.info["HostConfig"]["ReadonlyRootfs"]
        assert (
            native["uid"] == 10001 and not native["mlflow_installed"] and not native["forbidden"]
        )
        assert native["prefix"] == native["base_prefix"] == "/usr/local"
        allowed = {
            "requirements-linux-py313.txt",
            "models/pipeline.joblib",
            "references/model_release.json",
            "examples/predict_request.json",
            "examples/predict_request_final.json",
            "mibici_demand/__init__.py",
            "mibici_demand/api.py",
            "mibici_demand/config.py",
            "mibici_demand/features.py",
            "mibici_demand/release.py",
            "mibici_demand/modeling/__init__.py",
            "mibici_demand/modeling/pipeline.py",
        }
        assert set(native["files"]) == allowed
        for name, digest in native["files"].items():
            host = default_model_path() if name == "models/pipeline.joblib" else config.ROOT / name
            assert sha256(host.read_bytes()).hexdigest() == digest
        result = {
            "estado": "verificado",
            "release": release,
            "image_id": linux.info["Image"],
            "image": IMAGE,
            "docker_run": linux.command,
            "status": status,
            "python_windows": direct.tolist(),
            "windows_api": responses[0],
            "docker_api": responses[1],
            "docker_native": native,
            "diferencia_maxima_windows_api": float(np.max(np.abs(values(responses[0]) - direct))),
            "diferencia_maxima_docker_api": float(np.max(np.abs(values(responses[1]) - direct))),
            "diferencia_maxima_docker_native": float(
                np.max(np.abs(np.asarray(native["predictions"]) - direct))
            ),
        }
        write_json(out / "inferencia_final.json", result)
        write_json(config.ROOT / "examples/predict_response_final.json", responses[0])
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in [
                        "estado",
                        "image_id",
                        "python_windows",
                        "diferencia_maxima_docker_api",
                    ]
                }
            )
        )
        return result


if __name__ == "__main__":
    run()

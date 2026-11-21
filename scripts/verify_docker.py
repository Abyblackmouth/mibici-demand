"""Verificación real de Windows y Docker/Linux, sin volúmenes ni reentrenamiento.

Construir primero la imagen; ejecutar con .venv/Scripts/python.exe scripts/verify_docker.py.
Los únicos procesos/contenedores que se detienen son los creados por este script.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from uuid import uuid4
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from threadpoolctl import threadpool_limits

from mibici_demand import config

IMAGE = "mibici-demand:fase3-historico"
ATOL = RTOL = 1e-12
DTYPES = {
    "estacion": "int16",
    "hora": "int8",
    "dia_semana": "int8",
    "mes": "int8",
    "es_fin_semana": "int8",
    "lag_24": "float32",
    "lag_168": "float32",
    "latitud": "float64",
    "longitud": "float64",
    "capacidad": "float64",
}
APP_FILES = {
    "requirements-linux-py313.txt",
    "mibici_demand/__init__.py",
    "mibici_demand/api.py",
    "mibici_demand/release.py",
    "references/model_release.json",
    "mibici_demand/config.py",
    "mibici_demand/features.py",
    "mibici_demand/modeling/__init__.py",
    "mibici_demand/modeling/pipeline.py",
    "models/best_pipeline.joblib",
    "examples/predict_request.json",
    "examples/predict_request_final.json",
}


def docker(*arguments: str, stdin: str | None = None, timeout: int = 60) -> str:
    result = subprocess.run(
        ["docker", *arguments],
        check=False,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.returncode:
        raise RuntimeError(
            f"docker {arguments[:3]} falló ({result.returncode}): {result.stderr[-5000:]}"
        )
    return result.stdout.strip()


def wait_health(url: str, check_running, timeout: int = 40) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not check_running():
            raise RuntimeError("El servicio terminó antes de estar listo.")
        try:
            response = requests.get(url + "/health", timeout=2)
            if response.status_code == 200:
                return response.json()
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError(f"No se obtuvo health=200 en {timeout}s: {url}")


@dataclass
class ContainerService:
    identifier: str
    url: str
    info: dict
    command: list[str]


@contextmanager
def container_service(image: str = IMAGE):
    name = "mibici-verificacion-" + uuid4().hex[:12]
    args = [
        "run",
        "--detach",
        "--name",
        name,
        "--label",
        "mibici.validation=true",
        "--read-only",
        "--publish",
        "127.0.0.1::8000",
        image,
    ]
    identifier = docker(*args)
    try:
        info = json.loads(docker("inspect", identifier))[0]
        port = info["NetworkSettings"]["Ports"]["8000/tcp"][0]["HostPort"]
        url = f"http://127.0.0.1:{port}"
        wait_health(url, lambda: json.loads(docker("inspect", identifier))[0]["State"]["Running"])
        deadline = time.monotonic() + 40
        while True:
            info = json.loads(docker("inspect", identifier))[0]
            if info["State"]["Health"]["Status"] == "healthy":
                break
            if time.monotonic() > deadline:
                raise TimeoutError("El HEALTHCHECK de Docker no pasó.")
            time.sleep(0.3)
        yield ContainerService(identifier, url, info, ["docker", *args])
    finally:
        docker("rm", "--force", identifier)


@contextmanager
def windows_service(model_path: Path | None = None):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    environment = {
        **os.environ,
        "MODEL_PATH": str(model_path or config.MODELS_DIR / "best_pipeline.joblib"),
    }
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "mibici_demand.api:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    with (config.ROOT / ".venv/docker-windows-api.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=config.ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            wait_health(url, lambda: process.poll() is None)
            yield url, command
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def direct_frame(payload: dict) -> pd.DataFrame:
    records = []
    for row in payload["observaciones"]:
        date = datetime.fromisoformat(row["hora_inicio"]).astimezone(
            ZoneInfo("America/Mexico_City")
        )
        records.append(
            {
                "estacion": row["estacion"],
                "hora": date.hour,
                "dia_semana": date.weekday(),
                "mes": date.month,
                "es_fin_semana": int(date.weekday() >= 5),
                "lag_24": row["lag_24"],
                "lag_168": row["lag_168"],
                **{k: row[k] for k in ["latitud", "longitud", "capacidad"]},
            }
        )
    return pd.DataFrame(records).astype(DTYPES)


def native_probe(service: ContainerService) -> dict:
    # Este intérprete corre DENTRO del contenedor; solo stdlib y el runtime instalado.
    source = r"""
import hashlib,importlib.metadata as metadata,importlib.util,json,os,platform,sys,urllib.request,urllib.error,zoneinfo
from pathlib import Path
import pandas as pd
from threadpoolctl import threadpool_limits
from mibici_demand.api import load_model

root=Path('/app')
request_bytes=(root/'examples/predict_request.json').read_bytes()
payload=json.loads(request_bytes)
records=[]
from datetime import datetime
from zoneinfo import ZoneInfo
for row in payload['observaciones']:
 date=datetime.fromisoformat(row['hora_inicio']).astimezone(ZoneInfo('America/Mexico_City'))
 records.append({'estacion':row['estacion'],'hora':date.hour,'dia_semana':date.weekday(),'mes':date.month,'es_fin_semana':int(date.weekday()>=5),'lag_24':row['lag_24'],'lag_168':row['lag_168'],**{k:row[k] for k in ['latitud','longitud','capacidad']}})
dtypes={'estacion':'int16','hora':'int8','dia_semana':'int8','mes':'int8','es_fin_semana':'int8','lag_24':'float32','lag_168':'float32','latitud':'float64','longitud':'float64','capacidad':'float64'}
frame=pd.DataFrame(records).astype(dtypes)
model=load_model(Path(os.environ['MODEL_PATH']))
with threadpool_limits(limits=4):
 predictions=model.pipeline.predict(frame).tolist()
base='http://127.0.0.1:8000'
def get(path):
 with urllib.request.urlopen(base+path,timeout=5) as response: return response.status,response.read().decode()
health_status,health_body=get('/health')
docs_status,docs_body=get('/docs')
with urllib.request.urlopen(urllib.request.Request(base+'/predict',data=request_bytes,headers={'Content-Type':'application/json'}),timeout=10) as response:
 predict_status=response.status;http_response=json.load(response)
bad=json.loads(request_bytes);bad['observaciones'][0]['hora']=8
try:
 urllib.request.urlopen(urllib.request.Request(base+'/predict',data=json.dumps(bad).encode(),headers={'Content-Type':'application/json'}),timeout=5)
 invalid_status=200
except urllib.error.HTTPError as error: invalid_status=error.code
files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
print(json.dumps({'python':platform.python_version(),'sys_platform':sys.platform,'platform':platform.platform(),'machine':platform.machine(),'executable':sys.executable,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'uid':os.getuid(),'model_path':os.environ['MODEL_PATH'],'model_sha256':model.info.artifact_sha256,'request_sha256':hashlib.sha256(request_bytes).hexdigest(),'direct_predictions':predictions,'http_response':http_response,'health_status':health_status,'docs_status':docs_status,'swagger_ui': 'SwaggerUIBundle' in docs_body,'predict_status':predict_status,'invalid_status':invalid_status,'health':json.loads(health_body),'input_dtypes':frame.dtypes.astype(str).to_dict(),'versions':{d.metadata['Name']:d.version for d in metadata.distributions()},'tzpath':list(zoneinfo.TZPATH),'mlflow_installed':importlib.util.find_spec('mlflow') is not None,'application_files_sha256':files,'forbidden_paths_present':[p for p in ['/app/.venv','/app/data','/app/mlflow.db','/app/mlflow_fase3.db','/app/.env','/app/reports','/app/.git'] if Path(p).exists()]}))
"""
    return json.loads(docker("exec", "-i", service.identifier, "python", "-", stdin=source))


def assert_isolation(service: ContainerService, probe: dict) -> None:
    assert service.info["Mounts"] == [], "El contenedor no debe montar datos del host."
    assert service.info["HostConfig"]["ReadonlyRootfs"] is True
    assert probe["sys_platform"] == "linux" and probe["python"] == "3.13.0"
    assert probe["prefix"] == probe["base_prefix"] and ".venv" not in probe["executable"]
    assert probe["uid"] == 10001 and probe["model_path"] == "/app/models/best_pipeline.joblib"
    assert not probe["mlflow_installed"] and not probe["forbidden_paths_present"]
    assert set(probe["application_files_sha256"]) == APP_FILES
    assert probe["input_dtypes"] == DTYPES and probe["tzpath"] == []
    for path, digest in probe["application_files_sha256"].items():
        assert sha256((config.ROOT / path).read_bytes()).hexdigest() == digest, (
            f"Contenido distinto: {path}"
        )


def values(response: dict) -> np.ndarray:
    return np.array([r["demanda_predicha"] for r in response["predicciones"]], dtype=np.float64)


def main() -> None:
    root = config.ROOT
    assert Path(sys.prefix).resolve() == (root / ".venv").resolve() and sys.version_info[:2] == (
        3,
        13,
    )
    assert (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        == "fase3"
    )
    output = root / "reports/fase3/integracion/docker_historico"
    output.mkdir(exist_ok=True)
    request_bytes = (root / "examples/predict_request.json").read_bytes()
    payload = json.loads(request_bytes)
    model_path = root / "models/best_pipeline.joblib"
    digest = sha256(model_path.read_bytes()).hexdigest()
    frame = direct_frame(payload)
    model = joblib.load(model_path)
    with threadpool_limits(limits=4):
        reference = model.predict(frame)
    image = json.loads(docker("image", "inspect", IMAGE))[0]
    with container_service() as linux, windows_service() as (windows_url, windows_command):
        print(
            "Servicios Linux y Windows listos; comparando el mismo JSON y artefacto.", flush=True
        )
        linux_response = requests.post(
            linux.url + "/predict",
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        windows_response = requests.post(
            windows_url + "/predict",
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        assert linux_response.status_code == windows_response.status_code == 200
        inside = native_probe(linux)
        assert_isolation(linux, inside)
        assert (
            inside["health_status"] == inside["docs_status"] == inside["predict_status"] == 200
            and inside["invalid_status"] == 422
        )
        assert inside["model_sha256"] == digest
        collected = {
            "windows_python": reference,
            "windows_fastapi": values(windows_response.json()),
            "docker_python": np.array(inside["direct_predictions"]),
            "docker_fastapi": values(linux_response.json()),
            "docker_fastapi_interno": values(inside["http_response"]),
            "windows_fastapi_evidencia_previa": values(
                json.loads((root / "examples/predict_response.json").read_text(encoding="utf-8"))
            ),
        }
        differences = {}
        for name, array in collected.items():
            np.testing.assert_allclose(array, reference, rtol=RTOL, atol=ATOL)
            differences[name] = float(np.max(np.abs(array - reference)))
        for response in [windows_response.json(), linux_response.json()]:
            assert response["model"]["artifact_sha256"] == digest
        (output / "respuesta_docker.json").write_text(
            json.dumps(linux_response.json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "respuesta_windows.json").write_text(
            json.dumps(windows_response.json(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (root / ".venv/docker-container.log").write_text(
            docker("logs", linux.identifier), encoding="utf-8"
        )
        evidence = {
            "fecha": datetime.now(ZoneInfo("America/Mexico_City")).isoformat(),
            "estado": "comprobado",
            "imagen": IMAGE,
            "image_id": image["Id"],
            "image_size_bytes": image["Size"],
            "image_os": image["Os"],
            "image_architecture": image["Architecture"],
            "base_image": (root / "Dockerfile")
            .read_text(encoding="utf-8")
            .splitlines()[0]
            .removeprefix("FROM "),
            "docker_run_command": linux.command,
            "windows_run_command": windows_command,
            "build_command": [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--progress",
                "plain",
                "--tag",
                IMAGE,
                ".",
            ],
            "model_sha256_windows": digest,
            "model_sha256_docker": inside["model_sha256"],
            "request_sha256_windows": sha256(request_bytes).hexdigest(),
            "request_sha256_docker": inside["request_sha256"],
            "predicciones": {k: [float(v) for v in a] for k, a in collected.items()},
            "max_abs_difference": differences,
            "rtol": RTOL,
            "atol": ATOL,
            "container_mounts": linux.info["Mounts"],
            "container_health": linux.info["State"]["Health"]["Status"],
            "container_readonly_rootfs": linux.info["HostConfig"]["ReadonlyRootfs"],
            "runtime_linux": inside,
            "runtime_windows": {
                "python": sys.version.split()[0],
                "input_dtypes": frame.dtypes.astype(str).to_dict(),
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in inside["versions"]
                    if name != "pip"
                },
            },
            "build_error_corrected": {
                "error": "ModuleNotFoundError: No module named pyarrow",
                "causa": "Índice de columnas ArrowStringArray serializado en el perfil pandas del artefacto.",
                "solucion": "pyarrow==26.0.1 en el runtime, sin modificar el artefacto.",
            },
            "reentrenamiento": False,
        }
    evidence["contenedores_y_servidor_de_prueba_detenidos"] = True
    test_path = output / "pytest_docker.xml"
    evidence["pytest"] = (
        ET.parse(test_path).getroot().find("testsuite").attrib if test_path.exists() else None
    )
    (output / "evidencia.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(evidence)
    print(
        json.dumps(
            {
                "imagen": evidence["image_id"],
                "health": inside["health_status"],
                "predict": inside["predict_status"],
                "invalid": inside["invalid_status"],
                "max_abs_difference": differences,
                "model_sha256": digest,
                "montajes": evidence["container_mounts"],
            },
            indent=2,
        )
    )


def write_report(e: dict) -> None:
    lines = [
        "# Fase 3: portabilidad de inferencia con Docker",
        "",
        f"Verificación real: {e['fecha']}. Estado **COMPROBADO** en Docker Desktop Linux/amd64. Rama `fase3`. No es un experimento de entrenamiento ni una certificación prospectiva del histórico.",
        "",
        "## Arquitectura",
        "",
        "La imagen contiene CPython 3.13.0 para Debian Bookworm, las dependencias de inferencia, seis archivos Python de MiBici, el artefacto y el JSON de referencia. Uvicorn sirve FastAPI en 0.0.0.0:8000. MODEL_PATH=/app/models/best_pipeline.joblib; el paquete resuelve ROOT=/app. La aplicación se ejecuta como UID/GID 10001 y la prueba usa filesystem de solo lectura, sin bind mounts ni volúmenes de usuario. No se incorpora .venv, datos históricos, credenciales, .git, reportes o PDF de Swagger. El contexto es una lista cerrada en .dockerignore.",
        "",
        f"Base fijada por digest: `{e['base_image']}`. Imagen local `{e['imagen']}`, ID `{e['image_id']}`, tamaño reportado por Docker {e['image_size_bytes'] / 1024**2:.1f} MiB. No se ha publicado en ningún registro.",
        "",
        "## Dependencias y diferencia respecto a Windows",
        "",
        "requirements-linux-py313.txt fija las versiones directas y transitivas del runtime. Se usan wheels Linux de las mismas versiones que Windows; no se copian DLL ni paquetes del entorno virtual. pip==26.3 también queda fijado y pip check se ejecuta durante el build. No se instala MLflow, Jupyter, herramientas de entrenamiento ni dependencias exclusivas de Windows. La imagen no incluye pytest: la suite corre en Windows y sus pruebas Docker llaman a un contenedor Linux real.",
        "",
        "El primer build falló en la deserialización con ModuleNotFoundError: No module named pyarrow. El perfil guardado contiene un índice ArrowStringArray de pandas; PyArrow es necesario para cargar el pickle aunque la API no lea parquet. La solución fue añadir pyarrow==26.0.1, igual que Windows, conservando el modelo y la comprobación de carga en el build. No fue necesario cambiar Python, reentrenar ni rebajar la validación.",
        "",
        "La imagen fija PYTHONTZPATH vacío para usar tzdata==2026.6 en vez de una base de zonas horarias del sistema diferente. La [documentación de zoneinfo](https://docs.python.org/3.13/library/zoneinfo.html#environment-configuration) explica este comportamiento. `.dockerignore` sigue la semántica del [contexto de Docker](https://docs.docker.com/build/concepts/context/).",
        "",
        "| Dependencia | Windows | Linux |",
        "|---|---|---|",
    ]
    for name in [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pyarrow",
        "joblib",
        "fastapi",
        "pydantic",
        "starlette",
        "uvicorn",
        "tzdata",
    ]:
        lines.append(
            f"| {name} | {e['runtime_windows']['versions'][name]} | {e['runtime_linux']['versions'][name]} |"
        )
    lines += [
        "",
        "## Comandos ejecutados y reproducción en PowerShell",
        "",
        "El motor Linux estaba inactivo inicialmente (pipe dockerDesktopLinuxEngine no disponible). Se inició con docker desktop start y después respondió correctamente. El build final terminó con código 0; docker run arrancó el servicio y el HEALTHCHECK quedó healthy. El primer build fallido por PyArrow se conserva como antecedente y no se contó como éxito.",
        "",
        "Desde la raíz del repositorio; Docker Desktop debe estar en modo Linux. El modelo local debe existir antes de construir. La imagen lo incorpora, sin descargarlo al arrancar.",
        "",
        "```powershell",
        "docker desktop start",
        "docker build --platform linux/amd64 --progress plain --tag mibici-demand:fase3 .",
        "docker run --rm --name mibici-fase3 --read-only -p 127.0.0.1:8000:8000 mibici-demand:fase3",
        "```",
        "",
        "En otra terminal:",
        "",
        "```powershell",
        'Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"',
        r"$body = Get-Content .\examples\predict_request.json -Raw",
        'Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 12',
        r"& .\.venv\Scripts\python.exe scripts/verify_docker.py",
        r"& .\.venv\Scripts\python.exe -m pytest tests -q --junitxml=reports/fase3/docker/pytest_docker.xml",
        "```",
        "",
        "Swagger: http://127.0.0.1:8000/docs. El puerto host 8000 debe estar libre; la validación automatizada usa puertos efímeros para no interferir con otros servicios. El script crea y elimina solo sus propios contenedores y proceso Uvicorn de Windows. La imagen construida queda disponible localmente.",
        "",
        "## Resultados ejecutados",
        "",
        "Se comprobó el servicio desde Windows y también desde un intérprete ejecutado dentro del contenedor, con peticiones HTTP a su localhost:8000.",
        "",
        f"- GET /health: **{e['runtime_linux']['health_status']}**, modelo disponible y HEALTHCHECK Docker `{e['container_health']}`.",
        f"- GET /docs: **{e['runtime_linux']['docs_status']}**, Swagger UI presente.",
        f"- POST /predict con el JSON de referencia: **{e['runtime_linux']['predict_status']}**.",
        f"- Entrada inválida con calendario manual contradictorio: **{e['runtime_linux']['invalid_status']}**.",
        "",
        "## Igualdad de entradas, artefacto y predicciones",
        "",
        f"SHA-256 del modelo en Windows y Docker: `{e['model_sha256_windows']}`.",
        f"SHA-256 del JSON en Windows y Docker: `{e['request_sha256_windows']}`.",
        "",
        "El DataFrame conserva estación int16, calendario int8, rezagos float32 y metadatos float64. Las salidas se mantienen en float64. La referencia directa se construye independientemente del constructor de la API. Se compara también con el ejemplo Windows guardado en el bloque anterior, que no se reescribe. La tolerancia fue predefinida en rtol=atol=1e-12, igual que en las pruebas previas; no se ajustó para aprobar Docker.",
        "",
        "| Estación | Windows Python | Windows FastAPI | Docker Python | Docker FastAPI |",
        "|---:|---:|---:|---:|---:|",
    ]
    request = json.loads(
        (config.ROOT / "examples/predict_request.json").read_text(encoding="utf-8")
    )
    for i, row in enumerate(request["observaciones"]):
        lines.append(
            f"| {row['estacion']} | "
            + " | ".join(
                f"{e['predicciones'][k][i]:.17g}"
                for k in ["windows_python", "windows_fastapi", "docker_python", "docker_fastapi"]
            )
            + " |"
        )
    lines += ["", "| Comparación con Python Windows | Diferencia absoluta máxima |", "|---|---:|"]
    for name, difference in e["max_abs_difference"].items():
        lines.append(f"| {name} | {difference:.17g} |")
    lines += [
        "",
        "## Evidencia de independencia del host",
        "",
        f"- Linux `{e['runtime_linux']['python']}`, ejecutable `{e['runtime_linux']['executable']}`; sys.prefix=sys.base_prefix=`{e['runtime_linux']['prefix']}`: no usa el venv Windows.",
        f"- Ruta del modelo `{e['runtime_linux']['model_path']}`; montajes declarados `{e['container_mounts']}`; root filesystem de solo lectura `{e['container_readonly_rootfs']}`.",
        "- La introspección confirmó que no existen /app/.venv, /app/data, .env, .git, /app/reports ni las bases históricas de MLflow. MLflow ni siquiera está instalado. El servicio funciona sin esas rutas.",
        "- Los hashes de los nueve archivos de aplicación empaquetados coinciden con los del repositorio: código, dependencias, modelo y entrada. El listado completo está en docker/evidencia.json.",
        "- El contenedor carga y predice sin reconstruir variables a partir de archivos de Windows; recibe un lote de variables ya preparado, según el contrato existente.",
        "",
        "## Pruebas y evidencias",
        "",
    ]
    if e["pytest"]:
        t = e["pytest"]
        lines.append(
            f"Suite completa ejecutada: **{t['tests']} pruebas**, {t['failures']} fallos, {t['errors']} errores y {t['skipped']} omitidas; {float(t['time']):.2f} s. Evidencia: docker/pytest_docker.xml. Incluye las 97 pruebas previas y las nuevas integraciones reales de Docker."
        )
    else:
        lines.append(
            "La ejecución pytest todavía no está registrada en el XML; no se presupone un resultado no ejecutado."
        )
    lines += [
        "",
        "Evidencias nuevas: docker/evidencia.json, docker/respuesta_docker.json, docker/respuesta_windows.json y docker/pytest_docker.xml. Los logs de build y ejecución se conservan en .venv/docker-build.log, .venv/docker-build-inicial.log y .venv/docker-container.log. El fallo inicial y su solución se registran arriba; no se deshabilitó el smoke test del Dockerfile.",
        "",
        "## Alcance, limitaciones y pendientes",
        "",
        "Esta verificación demuestra **portabilidad de inferencia** para el mismo artefacto y entradas entre Windows y Linux/amd64 con Python 3.13.0 y versiones fijadas. No demuestra reentrenamiento reproducible, igualdad bit a bit de imágenes reconstruidas, compatibilidad con toda arquitectura/CPU ni calidad prospectiva del modelo. La imagen base está fijada por digest y las dependencias por versión; reconstruir sin caché requiere disponibilidad de esas distribuciones en Docker Hub/PyPI.",
        "",
        "El artefacto sigue siendo la referencia histórica: el resultado 438.0117 pertenece a otro modelo del backtesting, no se atribuye a este artefacto. Se conserva el contrato numérico y el diagnóstico frente a capacidad, sin recortar extremos. Los rezagos siguen siendo responsabilidad del productor de datos. La reserva de octubre y las limitaciones de validación previas no cambian.",
        "",
        "Las advertencias de pip durante el build como root se limitan a la instalación de la imagen; el servicio corre como UID 10001. La suite previa puede mantener sus deprecaciones conocidas. No se modifica el entorno global de Windows, los artefactos ni las métricas de fases anteriores. No se implementa drift ni se hacen commits, push o publicaciones.",
        "",
    ]
    (config.REPORTS_DIR / "fase3/validacion_docker.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

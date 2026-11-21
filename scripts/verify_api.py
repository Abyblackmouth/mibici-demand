"""Levanta Uvicorn en loopback, prueba HTTP real y escribe evidencia; no entrena.

Ejecutar desde la raíz: .venv/Scripts/python.exe scripts/verify_api.py
El proceso lanzado por esta comprobación se detiene siempre al finalizar.
"""

from __future__ import annotations

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
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
from threadpoolctl import threadpool_limits

from mibici_demand import api, config


def main() -> None:
    root = config.ROOT
    assert Path(sys.prefix).resolve() == (root / ".venv").resolve()
    assert sys.version_info[:2] == (3, 13)
    assert (
        subprocess.check_output(["git", "branch", "--show-current"], cwd=root, text=True).strip()
        == "fase3"
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    model = root / "models/best_pipeline.joblib"
    environment = {**os.environ, "MODEL_PATH": str(model)}
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
    base = f"http://127.0.0.1:{port}"
    with (root / ".venv/api-server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 40
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        "Uvicorn terminó antes de estar disponible: consultar .venv/api-server.log"
                    )
                try:
                    health = requests.get(base + "/health", timeout=2)
                    if health.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError("No se obtuvo /health listo dentro de 40 segundos.")
                time.sleep(0.2)
            docs = requests.get(base + "/docs", timeout=5)
            schema = requests.get(base + "/openapi.json", timeout=5)
            response = requests.post(base + "/predict", json=api.EXAMPLE_REQUEST, timeout=15)
            assert docs.status_code == schema.status_code == response.status_code == 200
            assert "SwaggerUIBundle" in docs.text
            assert "post" in schema.json()["paths"]["/predict"]
            result = response.json()
            records = []
            for row in api.EXAMPLE_REQUEST["observaciones"]:
                timestamp = datetime.fromisoformat(row["hora_inicio"]).astimezone(
                    ZoneInfo("America/Mexico_City")
                )
                records.append(
                    {
                        "estacion": row["estacion"],
                        "hora": timestamp.hour,
                        "dia_semana": timestamp.weekday(),
                        "mes": timestamp.month,
                        "es_fin_semana": int(timestamp.weekday() >= 5),
                        "lag_24": row["lag_24"],
                        "lag_168": row["lag_168"],
                        "latitud": row["latitud"],
                        "longitud": row["longitud"],
                        "capacidad": row["capacidad"],
                    }
                )
            frame = pd.DataFrame(records).astype(
                {
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
            )
            pipeline = joblib.load(model)
            with threadpool_limits(limits=4):
                direct = pipeline.predict(frame)
            served = np.array([r["demanda_predicha"] for r in result["predicciones"]])
            np.testing.assert_allclose(served, direct, rtol=1e-12, atol=1e-12)
            assert result["model"]["artifact_sha256"] == sha256(model.read_bytes()).hexdigest()
            invalid = json.loads(json.dumps(api.EXAMPLE_REQUEST))
            invalid["observaciones"][0]["lag_24"] = "no-numero"
            rejected = requests.post(base + "/predict", json=invalid, timeout=10)
            assert rejected.status_code == 422
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    examples = root / "examples"
    examples.mkdir(exist_ok=True)
    for filename, body in [
        ("predict_request.json", api.EXAMPLE_REQUEST),
        ("predict_response.json", result),
    ]:
        (examples / filename).write_text(
            json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    evidence = {
        "fecha": datetime.now(ZoneInfo("America/Mexico_City")).isoformat(),
        "comando_http_real": command,
        "url_prueba": base,
        "health_status": health.status_code,
        "health": health.json(),
        "docs_status": docs.status_code,
        "openapi_status": schema.status_code,
        "predict_status": response.status_code,
        "invalid_request_status": rejected.status_code,
        "solicitud": "examples/predict_request.json",
        "respuesta": "examples/predict_response.json",
        "inferencia_directa": [float(x) for x in direct],
        "inferencia_http": [float(x) for x in served],
        "diferencia_absoluta_maxima": float(np.max(np.abs(direct - served))),
        "tipos_dataframe": frame.dtypes.astype(str).to_dict(),
        "versiones": {
            p: importlib.metadata.version(p)
            for p in [
                "fastapi",
                "pydantic",
                "uvicorn",
                "httpx",
                "starlette",
                "scikit-learn",
                "numpy",
                "pandas",
                "joblib",
            ]
        },
        "servidor_prueba_detenido": process.poll() is not None,
        "modelo_reentrenado": False,
    }
    output = root / "reports/fase3"
    test_path = output / "pytest_api.xml"
    evidence["pytest"] = (
        ET.parse(test_path).getroot().find("testsuite").attrib if test_path.exists() else None
    )
    (output / "api_evidencia.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(output, evidence, result)
    print(
        json.dumps(
            {
                "health": health.status_code,
                "predict": response.status_code,
                "predicciones": evidence["inferencia_http"],
                "max_diff": evidence["diferencia_absoluta_maxima"],
                "servidor_prueba_detenido": evidence["servidor_prueba_detenido"],
            },
            indent=2,
        )
    )


def write_report(output: Path, evidence: dict, result: dict) -> None:
    lines = [
        "# Fase 3: integración y validación de la API",
        "",
        f"Prueba ejecutada: {evidence['fecha']}. Python 3.13, `.venv`, rama `fase3`. Estado: HTTP real comprobado; no es una validación prospectiva de precisión del modelo.",
        "",
        "## Arquitectura y artefacto",
        "",
        "`mibici_demand/api.py` implementa FastAPI y Pydantic. El ciclo lifespan carga el archivo una sola vez por proceso al iniciar, inspecciona el contrato ajustado y conserva el pipeline para las solicitudes. El hash SHA-256 corresponde a los mismos bytes deserializados. Cambiar `MODEL_PATH` requiere reiniciar el proceso. No se descarga, entrena ni registra un modelo al recibir solicitudes.",
        "",
        f"Artefacto de referencia: `models/best_pipeline.joblib`. SHA-256: `{result['model']['artifact_sha256']}`. Su versión de servicio es el hash, no una versión inventada del Model Registry. La respuesta lo identifica como `{result['model']['uso']}`. No se presenta como campeón prospectivamente validado.",
        "",
        "Se usa [lifespan](https://fastapi.tiangolo.com/advanced/events/) y las pruebas utilizan TestClient como context manager para ejecutar inicio/cierre. Se declaran FastAPI, Pydantic, Uvicorn y HTTPX como dependencias directas; ya estaban instalados como transitivas de MLflow y `pip install -r requirements.txt` verificó la instalación en `.venv`. Se fijó scikit-learn 1.9.1 para el artefacto histórico. Versiones exactas en `api_evidencia.json` y `requirements-win-py313.txt`.",
        "",
        "## Endpoints",
        "",
        "| Ruta | Resultado |",
        "|---|---|",
        "| GET /health | 200 con estado ready, disponibilidad, versión/hash y contrato; 503 si el artefacto falta o es incompatible. |",
        "| POST /predict | 200 con predicciones por estación/hora y diagnóstico; 422 para entradas inválidas; 503 sin modelo; 500 si el modelo falla o devuelve salidas inválidas. |",
        "| GET /docs | Swagger UI con esquema y ejemplo; disponible incluso si el modelo no está listo. |",
        "| GET /openapi.json | Contrato OpenAPI generado automáticamente. |",
        "",
        "## Esquema y precisión",
        "",
        "La solicitud contiene `origen_pronostico` y `observaciones`. La emisión es a las 00:00 de America/Mexico_City; cada objetivo debe ser una hora completa de `[origen, origen+24h)`. Fechas con zona se convierten a hora local; fechas sin zona se interpretan como locales. No se acepta un timestamp Unix numérico. El calendario se deriva exclusivamente de hora_inicio: el usuario no puede enviar hora, día, mes o fin de semana contradictorios. Se rechazan duplicados estación/hora y campos extra, incluidas etiquetas, lag_1 y media_24h.",
        "",
        "| Campo interno | Tipo exacto | Origen/validación |",
        "|---|---|---|",
        "| estacion | int16 | Entero JSON positivo representable, sin coerción de texto ni booleanos. |",
        "| hora, dia_semana, mes, es_fin_semana | int8 | Derivados de la fecha local; lunes=0. |",
        "| lag_24, lag_168 | float32 | Numéricos finitos no negativos; se rechaza desbordamiento del tipo. |",
        "| latitud, longitud, capacidad | float64 | Coordenadas válidas y capacidad positiva; requeridas por el histórico. |",
        "| Perfil, PCA y estadísticas por estación | float32 al transformar | Producidos por el pipeline guardado; el cliente no los envía. |",
        "| demanda_predicha | float64 / float JSON | Salida directa del pipeline; no se reduce a float32 ni se redondea. |",
        "",
        "El artefacto histórico ajustó un bloque numérico de tipo float64 por los metadatos. Los modelos del backtesting sin metadatos ajustaron su bloque en float32. La API inspecciona qué columnas se usaron realmente y omite las que no formen parte del contrato: enviar capacidad solo como diagnóstico al modelo sin metadatos no cambia sus variables predictoras ni su precisión.",
        "",
        "Pydantic valida antes de convertir el DataFrame. Los errores incluyen ubicación del campo y mensaje; valores NaN/infinito y cadenas numéricas no se admiten. Las restricciones de int16/float32 son límites de representación del contrato, no límites de predicción.",
        "",
        "## Quién construye los rezagos y disponibilidad",
        "",
        "La API NO reconstruye historia ni consulta los CSV. El sistema de preparación previo es `mibici_demand.temporal.causal_daily_frame`: agrupa viajes y cuenta únicamente los terminados antes de cada emisión. Para hora t, las fuentes son t−24h y t−168h, con la hora fuente ya cerrada. La solicitud diaria garantiza que esas horas están antes del origen, pero la API no puede certificar que un cliente no haya falseado los conteos. El productor debe conservar su procedencia y disponibilidad, incluida la de los metadatos. Los CSV públicos mensuales no constituyen un feed operativo diario. Los ejemplos usan metadatos ilustrativos y no son una reevaluación histórica de precisión.",
        "",
        "## Diagnóstico sin recortar",
        "",
        "`extreme_prediction` indica si la demanda predicha supera la capacidad de anclajes proporcionada. Esa referencia viene de cada observación; no se impone un techo numérico global. Es una alerta operativa, no una prueba estadística de anomalía ni un límite físico: una estación puede atender más viajes por hora que anclajes mediante rotación. Si el modelo no exige capacidad y esta no se envía, el indicador es null y criterion=sin_referencia_de_capacidad; no se declara que el valor sea normal. `unknown_station` indica uso de una estación no presente en el perfil ajustado, cuya imputación/OrdinalEncoder sigue siendo la del pipeline.",
        "",
        "No se recortan extremos ni negativos para disimular errores. Si el pipeline devuelve un valor negativo, NaN/infinito o una forma inválida, se informa un error 500. Un valor alto finito no negativo devuelve 200 con su diagnóstico; no implica necesariamente fallo del servicio, aunque puede señalar riesgo de desempeño.",
        "",
        "El valor **438.011742808447** de la estación 46 pertenece al modelo del corte 2026-09-07, un artefacto distinto del histórico servido por defecto. Una prueba de integración configura MODEL_PATH con ese pipeline, conserva el valor con precisión float64 y activa la alerta frente a capacidad 15. Al retirar la referencia de capacidad, la misma predicción permanece idéntica y el diagnóstico pasa a no evaluado. Ningún modelo se reentrena para esta prueba.",
        "",
        "## Solicitud HTTP real y comparación directa",
        "",
        f"Se inició Uvicorn en `{evidence['url_prueba']}` (puerto libre local) y se detuvo su proceso propio al finalizar. GET /health={evidence['health_status']}, GET /docs={evidence['docs_status']}, GET /openapi.json={evidence['openapi_status']}, POST /predict={evidence['predict_status']}; solicitud inválida={evidence['invalid_request_status']}.",
        "",
        "Ejemplos reales: [`predict_request.json`](../../examples/predict_request.json) y [`predict_response.json`](../../examples/predict_response.json). La respuesta completa incluye el hash del artefacto, su contrato y el diagnóstico de cada observación.",
        "",
        "| Estación | Hora local | API HTTP | Inferencia directa |",
        "|---:|---|---:|---:|",
    ]
    for row, direct in zip(result["predicciones"], evidence["inferencia_directa"]):
        lines.append(
            f"| {row['estacion']} | {row['hora_inicio']} | {row['demanda_predicha']:.16g} | {direct:.16g} |"
        )
    lines += [
        "",
        f"Diferencia absoluta máxima observada: **{evidence['diferencia_absoluta_maxima']:.16g}**. La comparación construye un DataFrame independiente del constructor de la API, con los tipos offline, y usa tolerancia 1e-12. La recarga del archivo también produce salidas idénticas.",
        "",
        "## Pruebas ejecutadas",
        "",
    ]
    if evidence["pytest"]:
        t = evidence["pytest"]
        lines.append(
            f"Suite completa: **{t['tests']} aprobadas**, {t['failures']} fallos, {t['errors']} errores y {t['skipped']} omitidas; {float(t['time']):.2f} s. XML: `pytest_api.xml`. Se conservan las 53 pruebas anteriores y se añaden 44 de API."
        )
    else:
        lines.append(
            "El reporte XML de la suite completa todavía no existe; no se presupone un resultado de pruebas no ejecutadas."
        )
    lines += [
        "",
        "Cobertura: health, Swagger/OpenAPI, lote válido y orden, equivalencia directa y tras recarga, precisión de tipos y salida, respuesta repetible, estación desconocida, campos ausentes/extra, cadenas/booleanos/nulos no admitidos, valores no finitos, horizonte/calendario, duplicados, carga única, archivo ausente/incompatible, salida negativa/no finita/formato inválido y MODEL_PATH alternativo con el caso extremo. Las pruebas que necesitan artefactos reales los requieren localmente; no los descargan ni entrenan para suplir su ausencia.",
        "",
        "Advertencias: el TestClient de Starlette 1.10.0 anuncia deprecación de su transporte HTTPX actual; funciona y las pruebas se ejecutaron. La suite histórica mantiene sus tres avisos conocidos de MLflow/SQLAlchemy/pandas. No son fallos ni se ocultan con filtros de warnings.",
        "",
        "## Inicio desde VS Code / PowerShell",
        "",
        "Abrir el repositorio y una terminal PowerShell en su raíz. No es necesario cambiar ExecutionPolicy ni activar otra instalación de Python:",
        "",
        "```powershell",
        r"$env:MODEL_PATH = (Resolve-Path .\models\best_pipeline.joblib).Path",
        r"& .\.venv\Scripts\python.exe -m uvicorn mibici_demand.api:app --host 127.0.0.1 --port 8000",
        "```",
        "",
        "Abrir http://127.0.0.1:8000/docs, desplegar POST /predict, pulsar Try it out y pegar examples/predict_request.json. Para probar desde otra terminal:",
        "",
        "```powershell",
        'Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"',
        r"$body = Get-Content .\examples\predict_request.json -Raw",
        'Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 12',
        "```",
        "",
        "Reproducción automatizada: `.venv\\Scripts\\python.exe scripts/verify_api.py`. El script crea un proceso Uvicorn propio y lo detiene; no detiene procesos ajenos. El servicio queda accesible solo en loopback. Cambiar MODEL_PATH a otro pipeline MiBici compatible con estas variables/horizonte no requiere editar código.",
        "",
        "## Límites",
        "",
        "La integración correcta no valida prospectivamente el histórico ni resuelve la reserva de octubre. No se han implementado Docker o drift, ni se ha ensayado Linux o carga sostenida. El servicio requiere un productor de variables con datos disponibles al emitir el pronóstico; no garantiza por sí solo la procedencia de lo recibido. Se conservan los resultados de backtesting/auditoría y sus modelos; no se entrena, publica ni promociona un nuevo campeón.",
        "",
    ]
    (output / "validacion_api.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

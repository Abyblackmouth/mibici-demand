# Fase 3: portabilidad de inferencia con Docker

Verificación real: 2026-11-20T19:02:41.941391-06:00. Estado **COMPROBADO** en Docker Desktop Linux/amd64. Rama `fase3`. No es un experimento de entrenamiento ni una certificación prospectiva del histórico.

## Arquitectura

La imagen contiene CPython 3.13.0 para Debian Bookworm, las dependencias de inferencia, seis archivos Python de MiBici, el artefacto y el JSON de referencia. Uvicorn sirve FastAPI en 0.0.0.0:8000. MODEL_PATH=/app/models/best_pipeline.joblib; el paquete resuelve ROOT=/app. La aplicación se ejecuta como UID/GID 10001 y la prueba usa filesystem de solo lectura, sin bind mounts ni volúmenes de usuario. No se incorpora .venv, datos históricos, credenciales, .git, reportes o PDF de Swagger. El contexto es una lista cerrada en .dockerignore.

Base fijada por digest: `python:3.13.0-slim-bookworm@sha256:0de818129b26ed8f46fd772f540c80e277b67a28229531a1ba0fdacfaed19bcb`. Imagen local `mibici-demand:fase3`, ID `sha256:2e2764270388e5f444a577b5dc46252d9c2867b4890a5544a7f03b78589a0187`, tamaño reportado por Docker 885.3 MiB. No se ha publicado en ningún registro.

## Dependencias y diferencia respecto a Windows

requirements-linux-py313.txt fija las versiones directas y transitivas del runtime. Se usan wheels Linux de las mismas versiones que Windows; no se copian DLL ni paquetes del entorno virtual. pip==26.3 también queda fijado y pip check se ejecuta durante el build. No se instala MLflow, Jupyter, herramientas de entrenamiento ni dependencias exclusivas de Windows. La imagen no incluye pytest: la suite corre en Windows y sus pruebas Docker llaman a un contenedor Linux real.

El primer build falló en la deserialización con ModuleNotFoundError: No module named pyarrow. El perfil guardado contiene un índice ArrowStringArray de pandas; PyArrow es necesario para cargar el pickle aunque la API no lea parquet. La solución fue añadir pyarrow==26.0.1, igual que Windows, conservando el modelo y la comprobación de carga en el build. No fue necesario cambiar Python, reentrenar ni rebajar la validación.

La imagen fija PYTHONTZPATH vacío para usar tzdata==2026.6 en vez de una base de zonas horarias del sistema diferente. La [documentación de zoneinfo](https://docs.python.org/3.13/library/zoneinfo.html#environment-configuration) explica este comportamiento. `.dockerignore` sigue la semántica del [contexto de Docker](https://docs.docker.com/build/concepts/context/).

| Dependencia | Windows | Linux |
|---|---|---|
| numpy | 2.5.5 | 2.5.5 |
| pandas | 3.1.1 | 3.1.1 |
| scipy | 1.18.1 | 1.18.1 |
| scikit-learn | 1.9.1 | 1.9.1 |
| pyarrow | 26.0.1 | 26.0.1 |
| joblib | 1.6.0 | 1.6.0 |
| fastapi | 0.165.0 | 0.165.0 |
| pydantic | 2.14.4 | 2.14.4 |
| starlette | 1.10.0 | 1.10.0 |
| uvicorn | 0.54.0 | 0.54.0 |
| tzdata | 2026.6 | 2026.6 |

## Comandos ejecutados y reproducción en PowerShell

El motor Linux estaba inactivo inicialmente (pipe dockerDesktopLinuxEngine no disponible). Se inició con docker desktop start y después respondió correctamente. El build final terminó con código 0; docker run arrancó el servicio y el HEALTHCHECK quedó healthy. El primer build fallido por PyArrow se conserva como antecedente y no se contó como éxito.

Desde la raíz del repositorio; Docker Desktop debe estar en modo Linux. El modelo local debe existir antes de construir. La imagen lo incorpora, sin descargarlo al arrancar.

```powershell
docker desktop start
docker build --platform linux/amd64 --progress plain --tag mibici-demand:fase3 .
docker run --rm --name mibici-fase3 --read-only -p 127.0.0.1:8000:8000 mibici-demand:fase3
```

En otra terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
$body = Get-Content .\examples\predict_request.json -Raw
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 12
& .\.venv\Scripts\python.exe scripts/verify_docker.py
& .\.venv\Scripts\python.exe -m pytest tests -q --junitxml=reports/fase3/docker/pytest_docker.xml
```

Swagger: http://127.0.0.1:8000/docs. El puerto host 8000 debe estar libre; la validación automatizada usa puertos efímeros para no interferir con otros servicios. El script crea y elimina solo sus propios contenedores y proceso Uvicorn de Windows. La imagen construida queda disponible localmente.

## Resultados ejecutados

Se comprobó el servicio desde Windows y también desde un intérprete ejecutado dentro del contenedor, con peticiones HTTP a su localhost:8000.

- GET /health: **200**, modelo disponible y HEALTHCHECK Docker `healthy`.
- GET /docs: **200**, Swagger UI presente.
- POST /predict con el JSON de referencia: **200**.
- Entrada inválida con calendario manual contradictorio: **422**.

## Igualdad de entradas, artefacto y predicciones

SHA-256 del modelo en Windows y Docker: `0faf37a57e05346ebfd59e695320a1e33eb5cecea98f1aec46858992319ff6b2`.
SHA-256 del JSON en Windows y Docker: `e4ffd85033074ab6a6b863be74a2c48cd18b09ba698b422e628abf4a2575d237`.

El DataFrame conserva estación int16, calendario int8, rezagos float32 y metadatos float64. Las salidas se mantienen en float64. La referencia directa se construye independientemente del constructor de la API. Se compara también con el ejemplo Windows guardado en el bloque anterior, que no se reescribe. La tolerancia fue predefinida en rtol=atol=1e-12, igual que en las pruebas previas; no se ajustó para aprobar Docker.

| Estación | Windows Python | Windows FastAPI | Docker Python | Docker FastAPI |
|---:|---:|---:|---:|---:|
| 2 | 2.6798914860304888 | 2.6798914860304888 | 2.6798914860304888 | 2.6798914860304888 |
| 46 | 1.6550795827162177 | 1.6550795827162177 | 1.6550795827162177 | 1.6550795827162177 |
| 999 | 1.1398398001926293 | 1.1398398001926293 | 1.1398398001926293 | 1.1398398001926293 |

| Comparación con Python Windows | Diferencia absoluta máxima |
|---|---:|
| windows_python | 0 |
| windows_fastapi | 0 |
| docker_python | 0 |
| docker_fastapi | 0 |
| docker_fastapi_interno | 0 |
| windows_fastapi_evidencia_previa | 0 |

## Evidencia de independencia del host

- Linux `3.13.0`, ejecutable `/usr/local/bin/python`; sys.prefix=sys.base_prefix=`/usr/local`: no usa el venv Windows.
- Ruta del modelo `/app/models/best_pipeline.joblib`; montajes declarados `[]`; root filesystem de solo lectura `True`.
- La introspección confirmó que no existen /app/.venv, /app/data, .env, .git, /app/reports ni las bases históricas de MLflow. MLflow ni siquiera está instalado. El servicio funciona sin esas rutas.
- Los hashes de los nueve archivos de aplicación empaquetados coinciden con los del repositorio: código, dependencias, modelo y entrada. El listado completo está en docker/evidencia.json.
- El contenedor carga y predice sin reconstruir variables a partir de archivos de Windows; recibe un lote de variables ya preparado, según el contrato existente.

## Pruebas y evidencias

Suite completa ejecutada: **107 pruebas**, 0 fallos, 0 errores y 0 omitidas; 54.68 s. Evidencia: docker/pytest_docker.xml. Incluye las 97 pruebas previas y las nuevas integraciones reales de Docker.

Evidencias nuevas: docker/evidencia.json, docker/respuesta_docker.json, docker/respuesta_windows.json y docker/pytest_docker.xml. Los logs de build y ejecución se conservan en .venv/docker-build.log, .venv/docker-build-inicial.log y .venv/docker-container.log. El fallo inicial y su solución se registran arriba; no se deshabilitó el smoke test del Dockerfile.

## Alcance, limitaciones y pendientes

Esta verificación demuestra **portabilidad de inferencia** para el mismo artefacto y entradas entre Windows y Linux/amd64 con Python 3.13.0 y versiones fijadas. No demuestra reentrenamiento reproducible, igualdad bit a bit de imágenes reconstruidas, compatibilidad con toda arquitectura/CPU ni calidad prospectiva del modelo. La imagen base está fijada por digest y las dependencias por versión; reconstruir sin caché requiere disponibilidad de esas distribuciones en Docker Hub/PyPI.

El artefacto sigue siendo la referencia histórica: el resultado 438.0117 pertenece a otro modelo del backtesting, no se atribuye a este artefacto. Se conserva el contrato numérico y el diagnóstico frente a capacidad, sin recortar extremos. Los rezagos siguen siendo responsabilidad del productor de datos. La reserva de octubre y las limitaciones de validación previas no cambian.

Las advertencias de pip durante el build como root se limitan a la instalación de la imagen; el servicio corre como UID 10001. La suite previa puede mantener sus deprecaciones conocidas. No se modifica el entorno global de Windows, los artefactos ni las métricas de fases anteriores. No se implementa drift ni se hacen commits, push o publicaciones.

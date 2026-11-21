# Fase 3: integración y validación de la API

Prueba ejecutada: 2026-11-19T20:51:07.152290-06:00. Python 3.13, `.venv`, rama `fase3`. Estado: HTTP real comprobado; no es una validación prospectiva de precisión del modelo.

## Arquitectura y artefacto

`mibici_demand/api.py` implementa FastAPI y Pydantic. El ciclo lifespan carga el archivo una sola vez por proceso al iniciar, inspecciona el contrato ajustado y conserva el pipeline para las solicitudes. El hash SHA-256 corresponde a los mismos bytes deserializados. Cambiar `MODEL_PATH` requiere reiniciar el proceso. No se descarga, entrena ni registra un modelo al recibir solicitudes.

Artefacto de referencia: `models/best_pipeline.joblib`. SHA-256: `0faf37a57e05346ebfd59e695320a1e33eb5cecea98f1aec46858992319ff6b2`. Su versión de servicio es el hash, no una versión inventada del Model Registry. La respuesta lo identifica como `referencia_integracion_no_validado_prospectivamente`. No se presenta como campeón prospectivamente validado.

Se usa [lifespan](https://fastapi.tiangolo.com/advanced/events/) y las pruebas utilizan TestClient como context manager para ejecutar inicio/cierre. Se declaran FastAPI, Pydantic, Uvicorn y HTTPX como dependencias directas; ya estaban instalados como transitivas de MLflow y `pip install -r requirements.txt` verificó la instalación en `.venv`. Se fijó scikit-learn 1.9.1 para el artefacto histórico. Versiones exactas en `api_evidencia.json` y `requirements-win-py313.txt`.

## Endpoints

| Ruta | Resultado |
|---|---|
| GET /health | 200 con estado ready, disponibilidad, versión/hash y contrato; 503 si el artefacto falta o es incompatible. |
| POST /predict | 200 con predicciones por estación/hora y diagnóstico; 422 para entradas inválidas; 503 sin modelo; 500 si el modelo falla o devuelve salidas inválidas. |
| GET /docs | Swagger UI con esquema y ejemplo; disponible incluso si el modelo no está listo. |
| GET /openapi.json | Contrato OpenAPI generado automáticamente. |

## Esquema y precisión

La solicitud contiene `origen_pronostico` y `observaciones`. La emisión es a las 00:00 de America/Mexico_City; cada objetivo debe ser una hora completa de `[origen, origen+24h)`. Fechas con zona se convierten a hora local; fechas sin zona se interpretan como locales. No se acepta un timestamp Unix numérico. El calendario se deriva exclusivamente de hora_inicio: el usuario no puede enviar hora, día, mes o fin de semana contradictorios. Se rechazan duplicados estación/hora y campos extra, incluidas etiquetas, lag_1 y media_24h.

| Campo interno | Tipo exacto | Origen/validación |
|---|---|---|
| estacion | int16 | Entero JSON positivo representable, sin coerción de texto ni booleanos. |
| hora, dia_semana, mes, es_fin_semana | int8 | Derivados de la fecha local; lunes=0. |
| lag_24, lag_168 | float32 | Numéricos finitos no negativos; se rechaza desbordamiento del tipo. |
| latitud, longitud, capacidad | float64 | Coordenadas válidas y capacidad positiva; requeridas por el histórico. |
| Perfil, PCA y estadísticas por estación | float32 al transformar | Producidos por el pipeline guardado; el cliente no los envía. |
| demanda_predicha | float64 / float JSON | Salida directa del pipeline; no se reduce a float32 ni se redondea. |

El artefacto histórico ajustó un bloque numérico de tipo float64 por los metadatos. Los modelos del backtesting sin metadatos ajustaron su bloque en float32. La API inspecciona qué columnas se usaron realmente y omite las que no formen parte del contrato: enviar capacidad solo como diagnóstico al modelo sin metadatos no cambia sus variables predictoras ni su precisión.

Pydantic valida antes de convertir el DataFrame. Los errores incluyen ubicación del campo y mensaje; valores NaN/infinito y cadenas numéricas no se admiten. Las restricciones de int16/float32 son límites de representación del contrato, no límites de predicción.

## Quién construye los rezagos y disponibilidad

La API NO reconstruye historia ni consulta los CSV. El sistema de preparación previo es `mibici_demand.temporal.causal_daily_frame`: agrupa viajes y cuenta únicamente los terminados antes de cada emisión. Para hora t, las fuentes son t−24h y t−168h, con la hora fuente ya cerrada. La solicitud diaria garantiza que esas horas están antes del origen, pero la API no puede certificar que un cliente no haya falseado los conteos. El productor debe conservar su procedencia y disponibilidad, incluida la de los metadatos. Los CSV públicos mensuales no constituyen un feed operativo diario. Los ejemplos usan metadatos ilustrativos y no son una reevaluación histórica de precisión.

## Diagnóstico sin recortar

`extreme_prediction` indica si la demanda predicha supera la capacidad de anclajes proporcionada. Esa referencia viene de cada observación; no se impone un techo numérico global. Es una alerta operativa, no una prueba estadística de anomalía ni un límite físico: una estación puede atender más viajes por hora que anclajes mediante rotación. Si el modelo no exige capacidad y esta no se envía, el indicador es null y criterion=sin_referencia_de_capacidad; no se declara que el valor sea normal. `unknown_station` indica uso de una estación no presente en el perfil ajustado, cuya imputación/OrdinalEncoder sigue siendo la del pipeline.

No se recortan extremos ni negativos para disimular errores. Si el pipeline devuelve un valor negativo, NaN/infinito o una forma inválida, se informa un error 500. Un valor alto finito no negativo devuelve 200 con su diagnóstico; no implica necesariamente fallo del servicio, aunque puede señalar riesgo de desempeño.

El valor **438.011742808447** de la estación 46 pertenece al modelo del corte 2026-09-07, un artefacto distinto del histórico servido por defecto. Una prueba de integración configura MODEL_PATH con ese pipeline, conserva el valor con precisión float64 y activa la alerta frente a capacidad 15. Al retirar la referencia de capacidad, la misma predicción permanece idéntica y el diagnóstico pasa a no evaluado. Ningún modelo se reentrena para esta prueba.

## Solicitud HTTP real y comparación directa

Se inició Uvicorn en `http://127.0.0.1:52082` (puerto libre local) y se detuvo su proceso propio al finalizar. GET /health=200, GET /docs=200, GET /openapi.json=200, POST /predict=200; solicitud inválida=422.

Ejemplos reales: [`predict_request.json`](../../examples/predict_request.json) y [`predict_response.json`](../../examples/predict_response.json). La respuesta completa incluye el hash del artefacto, su contrato y el diagnóstico de cada observación.

| Estación | Hora local | API HTTP | Inferencia directa |
|---:|---|---:|---:|
| 2 | 2026-09-12T08:00:00-06:00 | 2.679891486030489 | 2.679891486030489 |
| 46 | 2026-09-12T21:00:00-06:00 | 1.655079582716218 | 1.655079582716218 |
| 999 | 2026-09-12T08:00:00-06:00 | 1.139839800192629 | 1.139839800192629 |

Diferencia absoluta máxima observada: **0**. La comparación construye un DataFrame independiente del constructor de la API, con los tipos offline, y usa tolerancia 1e-12. La recarga del archivo también produce salidas idénticas.

## Pruebas ejecutadas

Suite completa: **97 aprobadas**, 0 fallos, 0 errores y 0 omitidas; 42.28 s. XML: `pytest_api.xml`. Se conservan las 53 pruebas anteriores y se añaden 44 de API.

Cobertura: health, Swagger/OpenAPI, lote válido y orden, equivalencia directa y tras recarga, precisión de tipos y salida, respuesta repetible, estación desconocida, campos ausentes/extra, cadenas/booleanos/nulos no admitidos, valores no finitos, horizonte/calendario, duplicados, carga única, archivo ausente/incompatible, salida negativa/no finita/formato inválido y MODEL_PATH alternativo con el caso extremo. Las pruebas que necesitan artefactos reales los requieren localmente; no los descargan ni entrenan para suplir su ausencia.

Advertencias: el TestClient de Starlette 1.10.0 anuncia deprecación de su transporte HTTPX actual; funciona y las pruebas se ejecutaron. La suite histórica mantiene sus tres avisos conocidos de MLflow/SQLAlchemy/pandas. No son fallos ni se ocultan con filtros de warnings.

## Inicio desde VS Code / PowerShell

Abrir el repositorio y una terminal PowerShell en su raíz. No es necesario cambiar ExecutionPolicy ni activar otra instalación de Python:

```powershell
$env:MODEL_PATH = (Resolve-Path .\models\best_pipeline.joblib).Path
& .\.venv\Scripts\python.exe -m uvicorn mibici_demand.api:app --host 127.0.0.1 --port 8000
```

Abrir http://127.0.0.1:8000/docs, desplegar POST /predict, pulsar Try it out y pegar examples/predict_request.json. Para probar desde otra terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
$body = Get-Content .\examples\predict_request.json -Raw
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 12
```

Reproducción automatizada: `.venv\Scripts\python.exe scripts/verify_api.py`. El script crea un proceso Uvicorn propio y lo detiene; no detiene procesos ajenos. El servicio queda accesible solo en loopback. Cambiar MODEL_PATH a otro pipeline MiBici compatible con estas variables/horizonte no requiere editar código.

## Límites

La integración correcta no valida prospectivamente el histórico ni resuelve la reserva de octubre. No se han implementado Docker o drift, ni se ha ensayado Linux o carga sostenida. El servicio requiere un productor de variables con datos disponibles al emitir el pronóstico; no garantiza por sí solo la procedencia de lo recibido. Se conservan los resultados de backtesting/auditoría y sus modelos; no se entrena, publica ni promociona un nuevo campeón.

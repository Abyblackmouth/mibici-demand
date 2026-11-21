# Fase 3: integrar validación causal, serving portable y monitoreo trazable

La integración anterior servía el campeón histórico en API/Docker y monitoreaba otro modelo. Esta entrega unifica el candidato técnico `mibici-temporal-20260921-v1` mediante manifiesto y SHA-256, conserva el histórico y todas sus evidencias, y distingue despliegue académico de aprobación productiva.

El candidato es una copia exacta del último corte causal disponible por cronología, sin búsqueda ni selección por métricas de prueba. FastAPI declara procedencia, tipos y corte mínimo, y rechaza predicciones retrospectivas anteriores a su entrenamiento. Docker incluye el mismo artefacto y el monitoreo se ejecuta en su semana posterior, guardando los diez escenarios previos por separado.

Se añadió snapshot versionado, restauración de artefactos con hashes, reproducción real del entrenamiento de tres pipelines en Windows limpio y Linux, y registro independiente MLflow con alias `candidate`. Se corrigió la CV histórica por filas para dividir timestamps completos con purga y se aislaron salidas de nuevas ejecuciones para conservar resultados anteriores. La documentación integra arquitectura, instalación, resultados, límites, rúbrica, roles y publicación pendiente.

Validación ejecutada:

- 161 pruebas aprobadas, cero fallos/omisiones; se conservan las 140 anteriores y sus criterios.
- Ruff sobre paquete, scripts y pruebas, comprobación de formato, pip check y diff check.
- Cuatro métricas idénticas en los tres reentrenamientos limpios; diferencia máxima de predicciones Windows 0, Linux ≈3,6e-15 (tolerancia de métricas predeclarada 1e-6 absoluta/relativa).
- HTTP real Windows/Linux y Docker healthy: health/predict/docs 200, inválidos 422; diferencia máxima de serving 0 y SHA del artefacto idéntico.
- Revisión independiente de fuentes, métricas de cortes, SQLite original y escenarios, con conservación de evidencias anteriores.

Limitaciones para revisión: el candidato no tiene aprobación productiva; se conserva el extremo 438 del primer corte; 28 días y umbrales PSI no están calibrados como óptimos/SLO; WAPE depende de la mezcla; NaN se evalúa en el pipeline y la API lo rechaza. No está disponible el snapshot original para repetir exactamente la búsqueda de Fase 2. El paquete local de modelos/datos debe distribuirse como asset externo con URL real, y DockerHub sigue pendiente.

Inventario propuesto: `reports/fase3/integracion/archivos_github.txt`. No incluir venv, datos completos, modelos binarios, almacenes/bases MLflow, credenciales, cachés ni temporales. El ZIP externo tiene su manifiesto en `references/artifact_bundle.json`.

Este documento es una **propuesta local para revisión**. No se creó un PR, commit, push, merge ni publicación. El PDF final y las contribuciones personales deben completarse por el equipo con evidencias reales.

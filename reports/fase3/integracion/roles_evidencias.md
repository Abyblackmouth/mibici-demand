# Roles, interacción y contribuciones verificables

Las responsabilidades siguientes son **funciones técnicas**, no nombramientos oficiales de integrantes. Los nombres se reproducen tal como figuran en Git; no se infiere identidad a partir de una matrícula. La autoría de un commit acredita sus cambios, no que esa persona ejecutó todos los experimentos ni trabajó sola.

| Rol funcional | Actividad, entrega e interacción | Evidencia verificable | Atribución respaldada / brecha |
|---|---|---|---|
| Data Engineer | Descarga, manifiesto SHA-256, limpieza y snapshot causal; entrega variables al Data Scientist y contrato de disponibilidad al ML Engineer | `dataset.py`, `cleaning.py`, `features.py`, `temporal.py`, `data_snapshot.json`; commit `20f954b` | Héctor Abraham Covarrubias Martínez figura como autor del pipeline; no acredita autoría individual de los cambios locales de Fase 3 |
| Data Scientist | ML Canvas, EDA, comparación de modelos/baselines, partición temporal y análisis de extremos; comunica límites al operador y ML Engineer | `planteamiento_del_problema.md`, notebooks, `ad512b7`, `7dbccea`, `34e4350`, `62793b2` | Git acredita a Héctor en pipeline/modelado; a `A01796998` en notebook Fase 2/figuras EDA; a `Eduardo-A01025391` en diccionario/hallazgos. No se inventa quién ejecutó entrenamientos |
| ML Engineer | Pipeline, Tracking/Registry, hash del candidato, inferencia y monitoreo; recibe snapshot/variables y entrega artefacto al servicio | `pipeline.py`, `tracking.py`, `release.py`, manifiestos; `ad512b7`, `b69bf27` | Héctor figura en pipeline y seguimiento. API/Docker/drift locales tienen evidencia ejecutada, pero aún no tienen contribuciones personales acreditadas en Git |
| Software Engineer | Refactorización, interfaces, pruebas unitarias/integración, trazabilidad de entradas y errores; integra contratos de los roles anteriores | `tests/`, `api.py`, scripts; `ebeaa77`, `7288a2b`, `87878d2` | Héctor figura en pruebas; `cobianrebeca` en `references/refactorizacion.md`; Eduardo en documentación de comandos. No atribuir a esos commits código que no contienen |
| SRE / Operaciones | Locks, aislamiento, Docker, healthcheck, reproducibilidad, alertas y recuperación; colabora con ML Engineer en liberación y con Data Engineer en calidad | `Dockerfile`, `Dockerfile.train`, locks, `inferencia_final.json`, reproducciones limpias, `alertas_acciones.csv` | Evidencia técnica de la sesión y archivos locales sin commit. **No hay asignación individual verificable de Fase 3**; el equipo debe documentarla y respaldarla |

## Límites de autoría

`git log --all --format='%h %an %s'` y `git show --name-only <sha>` sustentan las filas. Otros commits, por ejemplo `4ad2e7f`, `b647e5e` o `581b8f5`, aparecen sin archivos en la inspección de cambios y **no se usan para acreditar entregables concretos**. Un nombre genérico “Add files via upload” solo acredita los archivos que efectivamente aparecen en el diff.

Fase 3 se desarrolló con automatización asistida en estas sesiones. Sus reportes, hashes y JUnit acreditan ejecución, no autoría humana individual ni acuerdo de roles. No se hicieron commits en esta auditoría. Para el PDF y la publicación, el equipo debe validar nombres/roles, señalar aportes reales por archivo/PR/commit y describir colaboración, revisión y decisiones. No repartir porcentajes, horas ni contribuciones iguales sin evidencia.

## Para alcanzar Excelente en el criterio de 20 puntos

1. Confirmar la asignación real de los cinco roles y sus responsabilidades.
2. Relacionar cada aporte real con commits, revisiones o evidencias verificables; distinguir implementación, ejecución, análisis y documentación.
3. Explicar la interacción: datos disponibles → experimento causal → registro → servicio → operación → revisión de calidad/desempeño.
4. Incorporar este cuadro al reporte con la validación del equipo, sin presentar tareas propuestas como ya realizadas.

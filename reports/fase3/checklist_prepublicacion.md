# Checklist de prepublicación — rama fase3

No se ejecutaron commits, push, merge, publicación ni generación del PDF. La preparación local no equivale a tener enlaces públicos recuperables.

## Verificado localmente

- [x] Candidato único por manifiesto y SHA-256 en API, Docker y monitoreo integrado; campeón histórico intacto y documentado como referencia.
- [x] Snapshot v1 con seis parquets, tipos/periodos/semilla/configuración y SHA-256.
- [x] Entrenamiento real en Windows limpio y Linux; cuatro métricas idénticas en los tres cortes.
- [x] HTTP real Windows/Linux: health/predict/docs 200, inválidos 422, diferencia numérica 0; contenedor healthy, no root, sin volúmenes ni MLflow.
- [x] 161 pruebas aprobadas, cero fallos/omisiones; Ruff explícito sobre paquete/scripts/tests y pip check.
- [x] Conservación de evidencias históricas, revisión de tres cortes y diez escenarios originales; diez escenarios nuevos del candidato en directorio separado.
- [x] MLflow original contrastado en lectura y registro nuevo `candidate`, sin cambiar champion.
- [x] README con índice, instalación, artefactos, entrenamiento, API, Docker, monitoreo, pruebas y límites.
- [x] Reglas Git para excluir venv, cachés, credenciales, pesos, snapshots, bases y almacenes MLflow. No se borran archivos locales para lograrlo.
- [x] ZIP local de 14 miembros preparado y restaurado en prueba; SHA-256 y validación por miembro; rechazo de sobrescritura de versiones distintas.

## Qué incorporar a GitHub cuando se autorice

La lista **exacta y ordenada** es [integracion/archivos_github.txt](integracion/archivos_github.txt), generada con archivos rastreados y no rastreados no ignorados, y validada sin binarios de modelos/datos/DB/ZIP. Incluye fuentes Python y pruebas, Dockerfiles, locks, README/docs, notebooks históricos, manifiestos JSON/CSV y evidencias ligeras (métricas, figuras, JSON y JUnit).

La cifra y hash del inventario están en `integracion/auditoria_comprobaciones.json`. No utilizar `git add .` sin revisar esa lista y `git diff`; el inventario constituye la propuesta concreta, no una operación de staging. Las rutas locales y nombres Git en las evidencias se conservan por trazabilidad; el equipo debe revisar su exposición antes de publicar. El detector de credenciales aplicado busca formatos de tokens GitHub/AWS y claves privadas; no certifica ausencia de todo tipo de secreto.

## Distribución separada de fuentes

| Artefacto | Versión / hash / recuperación | Estado |
|---|---|---|
| ZIP de modelos y snapshot | `artifacts/mibici-fase3-v1.zip`; 16.793.052 bytes; SHA-256 `755b11dcec416efb6f5666af2e0406e8c67db31f9cadb1ef837481d161e2585b`; miembros en `references/artifact_bundle.json` | Preparado y restauración local probada; **no publicado**, URL vacía |
| Candidato | `models/final/pipeline.joblib`; SHA-256 `e355fb092ca5c6d3f8498369001a803c9afc399b82bdd379ee61b902809af372` | Incluido en ZIP; mismo artefacto de serving/monitoreo |
| Campeón y tres modelos temporales | Rutas/hashes de manifiesto del ZIP y reportes originales | Conservar como referencias; sin promoción histórica |
| Snapshot causal y datos preparados | Seis parquets y tres archivos preparados, hashes en manifiestos | Incluidos en ZIP; no contienen todos los años históricos |
| CSV crudos necesarios | Agosto/septiembre de 2026; URLs, tamaños y hashes en `data/raw/manifest.csv` | Recuperación de la fuente abierta, opcional copia separada del equipo; no subir al Git de fuentes |
| Imagen de inferencia | `mibici-demand:fase3`; ID local en `integracion/inferencia_final.json` | Build/run reales; **DockerHub pendiente** |
| MLflow local | `mlflow_integracion.db`, `mlflow_fase3.db`, `mlartifacts/` | Excluidos; copias de respaldo privadas si el equipo las necesita. Los manifiestos de serving evitan depender de estas bases |

Propuesta: adjuntar el ZIP y su manifiesto a una versión inmutable en GitHub Releases (o almacenamiento de artefactos autorizado), con versión/tag y SHA-256, y completar `download_url` solo con un enlace **real verificado**. [GitHub distingue los assets de una release del archivo de fuentes](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases). No confundir el ZIP de fuentes generado por GitHub con el paquete de modelos/datos. La imagen debe publicarse después bajo un namespace que confirme el equipo, conservando digest y etiqueta versionada; no se inventa ese namespace ni un enlace DockerHub.

## Pendiente antes de la entrega académica

- [ ] Cotejar transcripción de rúbrica con Canvas; matriz basada en la referencia proporcionada por el equipo.
- [ ] Equipo confirma nombres/roles, aportes reales, revisiones y evidencia de participación en Fase 3; no basta con asignar etiquetas de rol.
- [ ] Definir la licencia de código y confirmar condiciones de redistribución del snapshot antes de publicar: no existe un archivo LICENSE en esta copia y no se asigna una licencia automáticamente.
- [ ] Revisar diff/inventario, autorizar y ejecutar publicación GitHub y artefactos; luego probar recuperación desde un clon nuevo usando la URL publicada.
- [ ] Autorizar y ejecutar publicación DockerHub; registrar URL/digest reales y probar pull/run en entorno limpio. Es requisito obligatorio aún incumplido por restricción expresa.
- [ ] Redactar reflexión/conclusiones auténticas, integrar fases y gráficos/tablas contextualizados y comentarios del profesor en el PDF.
- [ ] Localizar/aportar `Soporte_Swagger.pdf` si se incorporará como evidencia visual; no se encontró en este repositorio.
- [ ] Generar posteriormente `Entrega_Final_Equipo07.pdf`, revisar enlaces y entregarlo en Canvas. Nada de esto se ejecuta en este bloque.

## Pendientes metodológicos que no se ocultan

No existe aprobación de producción ni holdout ciego certificado. Los 28 días no demostraron suficiencia frente a seis meses; no se optimizó esa ventana usando evaluación. La búsqueda histórica exacta de Fase 2 no es reproducible con el snapshot disponible. Las alertas son ilustrativas y la disponibilidad diaria se simula con CSV mensuales. Estos límites deben explicarse; no implican inventar resultados nuevos para obtener una rúbrica favorable.

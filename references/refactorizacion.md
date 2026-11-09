# De los notebooks a módulos

Cómo se reorganizó el notebook original (`Fase_1_Equipo_7.ipynb`, que trabajaba con un solo mes y una ruta de Google Drive):

| Notebook original | Ahora | Qué cambió |
|---|---|---|
| Celda 0: montar Drive y `os.chdir` | `config.py` | Rutas relativas a la raíz del repositorio; funciona en cualquier máquina |
| Celdas 1 a 4: `read_csv`, `info`, nulos | `dataset.read_trips` | Lectura como texto y conversión explícita; lo no convertible se cuenta |
| Celda 6: fechas, agrupar por estación y hora | `features.monthly_tables` | Por mes, para no cargar todo en memoria |
| Celdas 11 a 13: rejilla completa y rezagos | `features.build_station_hour_table` | Rejilla solo en la ventana activa de cada estación y en horas con servicio; se agregan pruebas de que no hay fuga |
| Celdas 17 a 20: histogramas, boxplots, correlación | `plots.py` | Funciones que guardan figuras; la estación se trata como categoría |
| Celdas 21 a 28: pruebas de nulos, duplicados y saltos | `cleaning.TripCleaner` y `tests/` | Reglas contadas sobre los viajes (donde viven los problemas reales) y pruebas automáticas |
| Imports de MLflow comentados | `tracking.py` y `cli registrar` | Cada versión de los datos queda registrada |

Se usa una clase (`TripCleaner`) porque la limpieza tiene configuración (perfil y umbrales) y estado de reporte; el resto son funciones sin estado.

## Fase 2: del análisis al modelado

El modelado se escribió directamente en módulos; los notebooks solo importan y muestran resultados.

| Lógica | Dónde vive | Qué cambió |
|---|---|---|
| Perfil de cada estación (demanda media, proporción en horas pico, componentes principales) | `features.StationProfileTransformer` | Reutiliza `station_profile_pca` de la Fase 1, pero se ajusta solo con los datos de entrenamiento, dentro del `Pipeline` |
| Variables válidas según el horizonte | `features.get_feature_columns` y `config.HORIZON_LAGS` | Con 24 horas solo se usan los rezagos de 24 y 168 horas |
| Preprocesamiento y estimador | `modeling/pipeline.py` (`build_model_pipeline`) | Un solo `Pipeline` para los cuatro algoritmos |
| Partición temporal, ajuste, evaluación y comparación | `modeling/train.py` | Métricas en `compute_metrics`; los espacios de búsqueda son constantes con nombre al inicio del módulo |
| Predicción por lotes | `modeling/predict.py` | Carga el modelo desde un archivo o desde el Model Registry y calcula la presión de redistribución |
| Registro de corridas y del campeón | `tracking.log_model_run` y `tracking.register_champion_model` | Cada corrida guarda parámetros, métricas, firma del modelo y archivos |
| Valores fijos | `config.py` | Semilla, días de validación y prueba, horas pico, horizonte, columnas de la estación y nombres del experimento y del modelo registrado |
| Puntos de entrada | `cli.py` (`entrenar`, `predecir`) y `Makefile` (`make train`, `make predict`) | Un comando por paso |
| Notebooks | `0.01` y `0.02` | Importan los módulos; no contienen lógica propia |


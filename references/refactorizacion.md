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

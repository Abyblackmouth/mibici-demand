# Diccionario de datos

## Archivos de viajes (`datos_abiertos_AAAA_MM.csv`, uno por mes)

Mismas 8 columnas verificadas en diciembre de 2014 y en septiembre de 2026. El encabezado de `Año_de_nacimiento` aparece de 5 formas entre los 142 archivos y tres meses publican las fechas como `DD/MM/AAAA h:mm`; la lectura normaliza ambos casos (ver `hallazgos_de_datos.md`, H1 y H2).

| Columna | Tipo tras la lectura | Descripción | Tratamiento |
|---|---|---|---|
| `Viaje_Id` | entero | Identificador del viaje | Se usa para detectar filas repetidas |
| `Usuario_Id` | entero | Identificador del usuario | No entra al modelo |
| `Genero` | texto (`M`, `F`, nulo) | Género | No entra al modelo; los nulos se conservan |
| `Año_de_nacimiento` | decimal | Año de nacimiento | Fuera de rango (antes de 1920 o menor de 10 años) pasa a nulo |
| `Inicio_del_viaje` | fecha y hora | Inicio del viaje | Define la hora a la que se asigna el viaje; obligatoria |
| `Fin_del_viaje` | fecha y hora | Fin del viaje | Se usa para validar la duración y la actividad de la estación destino |
| `Origen_Id` | entero | Estación de salida | Define la estación de la demanda; obligatoria |
| `Destino_Id` | entero | Estación de llegada | Se usa para la actividad de la estación |

## Catálogo de estaciones (`nomenclatura_AAAA_MM.csv`)

`id`, `name` (código de zona y calle; el prefijo `ex-` marca estaciones dadas de baja), `latitude`, `longitude`, `dpcapacity` (anclajes).

## Tabla de entrenamiento (`data/processed/demanda_estacion_hora_<perfil>.parquet`)

| Columna | Descripción |
|---|---|
| `estacion` | Estación (identificador, se trata como categoría) |
| `hora_inicio` | Hora a la que se agregan los viajes |
| `demanda` | **Objetivo**: viajes iniciados en la estación durante esa hora |
| `hora`, `dia_semana`, `mes`, `anio`, `es_fin_semana` | Calendario |
| `lag_1`, `lag_24`, `lag_168` | Demanda de la misma estación hace 1 hora, 1 día y 1 semana |
| `media_24h` | Media de las 24 horas anteriores (sin incluir la hora actual) |
| `latitud`, `longitud`, `capacidad` | Solo si el catálogo está en `data/raw` |

**Rezagos y horizonte.** Con el horizonte de las 24 horas del día siguiente solo se pueden usar `lag_24` y `lag_168`: `lag_1` y `media_24h` solo existen para la primera hora y quedan reservados para un horizonte de una hora.

# Hallazgos de los datos de MiBici

Registro de lo que se encontró al procesar los 142 archivos mensuales (diciembre de 2014 a septiembre de 2026) y de qué se hizo con cada cosa. Cada hallazgo indica su **evidencia**, su **tratamiento** y su **estado**. Las cifras salen de la ejecución del pipeline sobre los datos completos; lo que es una lectura de una gráfica o una interpretación se marca como tal.

## Resumen

| # | Hallazgo | Tratamiento | Estado |
|---|---|---|---|
| H1 | El encabezado del año de nacimiento aparece de 5 formas distintas | Normalización de los nombres de columna | Resuelto en código |
| H2 | Tres meses publican las fechas como `DD/MM/AAAA h:mm`, sin segundos | El formato se elige por mes, según donde caen los viajes | Resuelto en código |
| H3 | La limpieza elimina el 3.09% de las filas | Cinco reglas, cada una con su conteo | Resuelto en código |
| H4 | El umbral de 60 s del anclaje fallido es una decisión | Análisis de sensibilidad con un mes; se mantiene | Documentado |
| H5 | En los meses sin segundos la regla de 60 s es menos precisa | Se anota como limitación | Documentado |
| H6 | El contador de la página no coincide con la suma de los archivos | Sin explicación confirmada | Documentado |
| H7 | El horario de servicio se infiere de los datos | Umbral del 0.1% de los viajes del mes | Supuesto |
| H8 | La red de estaciones crece por escalones | La rejilla empieza cuando aparece cada estación | Resuelto en código |
| H9 | Caída por la pandemia en 2020 | Se conserva; se compara con la ventana desde 2022 | Documentado |
| H10 | Escalón de usuarios en enero de 2019 | Sin causa conocida | Documentado |
| H11 | Septiembre de 2026 es el mes con más viajes de toda la serie | Coincide con más estaciones activas | Documentado |
| H12 | La demanda está sesgada y el 43% son ceros | Los valores altos se conservan | Resuelto en código |
| H13 | El componente 1 del PCA es el tamaño de la estación | El componente 2 no se interpreta | Documentado |
| H14 | Unas 100 estaciones nuevas en 2026 tienen poca historia | Riesgo para la Fase 2 | Pendiente |
| H15 | Una estación aparece solo durante una hora y no tiene filas de entrenamiento | Queda en la rejilla sin filas finales | Documentado |

## H1. Cinco variantes del encabezado del año de nacimiento

- **Evidencia:** entre los 142 archivos hay 5 encabezados distintos: 78 sin comillas con `Año`, 21 con comillas y `Año`, 38 con comillas y la `ñ` dañada por la codificación, 3 sin comillas y con la `ñ` dañada, y 2 con otra forma dañada. Febrero de 2024 fue el primero que hizo fallar la lectura.
- **Tratamiento:** `dataset.read_trips` normaliza los nombres (minúsculas, sin acentos). Las columnas que no afectan la demanda (usuario, género y año de nacimiento) pueden faltar y quedan como nulas; las obligatorias (fecha de inicio, origen) hacen fallar la lectura con un mensaje que dice cuáles faltan.

## H2. Formato de fecha distinto en tres meses

- **Evidencia:** noviembre de 2023, septiembre de 2024 y octubre de 2024 usan `DD/MM/AAAA h:mm` (por ejemplo `01/11/2023 0:01`). La lectura automática las interpretaba como `MM/DD`, así que en noviembre de 2023 solo se leían los días del 1 al 12 (como enero a diciembre) y 233,833 viajes quedaban sin fecha: de 384,464 viajes quedaban 147,099.
- **Tratamiento:** `_parse_dates` elige entre la lectura automática y tres formatos alternativos el que deje los viajes dentro del mes del archivo. Con el cambio, noviembre de 2023 conserva 376,089 viajes. El resumen de limpieza guarda el formato usado por mes, y `procesar` avisa de cualquier mes con formato distinto, con fechas fuera de su mes o sin fecha válida. Un mes sin ningún viaje válido es un error.

## H3. Resultado de la limpieza

| Regla | Filas | % del total |
|---|---|---|
| Anclaje fallido (menos de 60 s, mismo origen y destino) | 1,205,672 | 3.085% |
| Duración imposible (negativa o de más de 24 h) | 1,832 | 0.005% |
| Filas duplicadas exactas | 758 | 0.002% |
| Sin fecha de inicio u origen | 0 | 0% |
| **Total eliminado** | **1,208,262** | **3.092%** |
| Año de nacimiento corregido (no elimina filas) | 4,572 | 0.012% |

De 39,078,984 filas originales quedan 37,870,722. Los 758 duplicados exactos están todos en septiembre de 2026, porque ese mes por sí solo tiene 758.

## H4. Umbral de 60 segundos

- **Evidencia (solo septiembre de 2026):** 33,812 de 490,056 viajes salen y llegan a la misma estación. Su duración se reparte así: menos de 30 s, 12,319; de 30 a 60 s, 5,339; de 60 a 90 s, 2,202; de 90 a 120 s, 925; de 120 a 180 s, 931; de 3 a 5 min, 1,247; de 5 a 10 min, 3,390; de 10 a 30 min, 6,726; más de 30 min, 733.
- **Lectura:** hay un valle entre 90 y 180 s y después vuelven a subir los viajes reales. Por eso 60 s es conservador: el tramo de 60 a 90 s parece contener todavía anclajes fallidos (interpretación, no dato). Subir el umbral a 120 s eliminaría unos 3,100 viajes más en ese mes (0.6%).
- **Decisión:** se mantiene 60 s. No se verificó en otros años. Cambiarlo exige volver a correr `procesar`, `construir` y `registrar`.

## H5. Meses sin segundos

Con marcas de tiempo al minuto, un viaje de 30 a 59 s cuya salida y llegada caen en minutos distintos aparece con 60 s y no se elimina. En una prueba con septiembre de 2026 convertido a ese formato se conservaron unos 7 mil viajes más (1.5%). Afecta a noviembre de 2023, septiembre de 2024 y octubre de 2024.

## H6. Contador de la página contra los archivos

La página de datos abiertos muestra 39,601,506 viajes; la suma de las filas de los 142 archivos es 39,078,984, una diferencia de 522,522 (1.3%). No viene de la lectura (cada archivo se lee completo). No hay una explicación confirmada.

## H7. Horario de servicio inferido

Una hora del día se considera sin servicio en un mes cuando concentra menos del 0.1% de los viajes de ese mes. Con esa regla, la tabla tiene 20 horas con servicio (la 0 y de la 5 a la 23) en 2015 y de 2021 a 2026, y 18 horas (de la 6 a la 23) en 2014 y de 2016 a 2020. No se contrastó con el horario oficial de cada año: una hora abierta con muy pocos viajes podría haber quedado excluida. Esas horas cuentan como ceros para los rezagos, pero no son filas de entrenamiento (5,650,555 filas en la tabla completa).

## H8. Red de estaciones por escalones

- **Evidencia (lectura de la gráfica, aproximada):** las estaciones con viajes pasan de unas 85 en 2014 a unas 115 en 2015, unas 235 desde agosto de 2016, unas 275 en 2018, unas 300 en 2021, unas 360 en 2023 y unas 468 en 2026. La tabla completa tiene 486 estaciones, porque incluye las que existieron en algún momento.
- **Tratamiento:** cada estación existe solo entre su primera y su última aparición (como origen o destino). Antes de abrir o después de cerrar no hay demanda cero, no hay demanda posible. Limitación: una estación cerrada temporalmente queda con ceros de ese lapso.

## H9. Pandemia

Los viajes limpios por mes pasan de 357,369 en febrero de 2020 a 288,176 en marzo y 124,951 en abril, y se recuperan poco a poco: 287,561 en diciembre de 2021 y 362,140 en marzo de 2022. Es una caída de demanda y no de red: las estaciones con viajes no bajan. Se conserva en los datos; la ventana desde 2022 la deja fuera y se compara con la historia completa en MLflow.

## H10. Escalón de usuarios en enero de 2019

| Mes | Viajes | Usuarios | Viajes por usuario | Estaciones | Duración mediana |
|---|---|---|---|---|---|
| Nov. 2018 | 301,923 | 14,346 | 21.0 | 275 | 9.8 min |
| Dic. 2018 | 278,244 | 14,405 | 19.3 | 273 | 9.8 min |
| Ene. 2019 | 393,718 | 17,394 | 22.6 | 274 | 10.0 min |
| Feb. 2019 | 378,747 | 17,418 | 21.7 | 273 | 10.0 min |

De diciembre a enero los viajes suben 41.5%, los usuarios 20.7% y los viajes por usuario 17.2%. Contra noviembre los viajes suben 30.4% y los usuarios 21.2%, porque diciembre es un mes bajo. En febrero los usuarios se quedan en el nuevo nivel. No cambian ni las estaciones ni la duración mediana. **No se conoce la causa**; puede ser un cambio de tarifa, una promoción o de cómo se cuentan los usuarios.

## H11. Septiembre de 2026

Tiene 471,655 viajes limpios, un 24.7% más que agosto (378,089) y el máximo de la serie. En esos meses las estaciones con viajes pasan de unas 370 a unas 468 (lectura de la gráfica, aproximada). Que ambas cosas coincidan no prueba que una cause la otra.

## H12. Distribución de la demanda

En la tabla completa, el 43.4% de las filas (estación y hora) no tiene viajes y la demanda media es 1.67. La valla del IQR queda en 5 viajes y el 7.0% de las filas la supera, concentrado en las horas pico: 17.9% de las filas de las 18:00, 12.5% de las 19:00, 12.0% de las 8:00, 11.5% de las 17:00 y 10.6% de las 14:00. Se interpreta como comportamiento real y se conserva; solo se eliminan los errores. La regla del IQR sirve poco con conteos muy sesgados y se usa para dimensionar el fenómeno.

## H13. PCA del perfil semanal de cada estación

El perfil de 168 valores (7 días por 24 horas) se reduce a 3 componentes que explican 76.9%, 8.2% y 3.8% de la varianza (88.8% en total). El componente 1 tiene una correlación de 0.997 con la demanda media de la estación: representa el tamaño de la estación. El componente 2 no se interpreta: su correlación con la proporción de demanda en horas pico entre semana es de solo 0.234.

## H14. Estaciones nuevas en 2026

Las cerca de 100 estaciones que aparecen en 2026 tienen pocos meses de historia, lo que dificulta predecir su demanda (arranque en frío) y la primera semana de cada estación se descarta por falta de rezagos. Es un riesgo para el modelado de la Fase 2.

## H15. Una estación sin filas de entrenamiento

- **Evidencia:** la estación 405 aparece una sola vez, el 8 de junio de 2026 de las 18:00 a las 19:00 (una hora de vida). Por eso la rejilla tiene 486 estaciones y la tabla final 485 (en la ventana desde 2022, 473 y 472).
- **Tratamiento:** al no tener ni una semana de historia, ninguna de sus filas llega a tener los rezagos completos y se descartan. No se interpreta la causa (podría ser una estación de prueba o un registro aislado).

## Dos versiones de la tabla de entrenamiento

| | Historia completa | Desde 2022 |
|---|---|---|
| Meses | 142 | 57 |
| Estaciones en la rejilla | 486 | 473 |
| Estaciones con filas de entrenamiento | 485 | 472 |
| Filas de la rejilla | 28,365,933 | 14,378,693 |
| Excluidas por horas sin servicio | 5,650,555 | 2,395,575 |
| Excluidas por falta de historial | 64,073 | 66,082 |
| **Filas finales** | **22,651,305** | **11,917,036** |
| Proporción de ceros | 43.4% | 42.8% |
| Demanda media | 1.67 | 1.70 |

El baseline estacional (la demanda de la misma hora de la semana anterior) se evalúa sobre la última semana de cada tabla, que es prácticamente la misma, así que sus métricas no distinguen una versión de otra. La comparación de ventanas será informativa cuando se entrenen modelos en la Fase 2.

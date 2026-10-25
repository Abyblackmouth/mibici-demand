# Planteamiento del problema y ML Canvas

Predicción de demanda de viajes por estación y hora con los datos abiertos de MiBici. Equipo 7, Fase 1.

## 1. Acercamiento inicial al problema

Partimos de lo trabajado en el curso: el planteamiento de problemas de ML con el Machine Learning Canvas, la limpieza y la exploración de datos (diagnóstico de calidad, valores inválidos, duplicados y atípicos), la estructura de proyecto con Cookiecutter Data Science, el refactoring de notebooks a módulos y el registro de experimentos y datasets con MLflow.

Del dominio partimos de un reto operativo típico de los sistemas de bicicletas compartidas con estaciones fijas: la demanda no se reparte de forma pareja en el espacio ni en el tiempo, de modo que algunas estaciones se quedan sin bicicletas mientras otras se quedan sin espacios libres. Nuestra relación con la bicicleta es recreativa; uno de nosotros la usa en el Parque Fundidora, en Monterrey. No somos usuarios de MiBici, así que partimos de los datos y no de la experiencia propia en el sistema. En un uso recreativo uno esperaría más actividad en fines de semana, y los datos muestran lo contrario: la demanda de MiBici es mayor de lunes a jueves, con picos a las 8:00 y a las 18:00, lo que sugiere un uso de movilidad cotidiana.

## 2. Contexto y datos

MiBici publica un archivo CSV por mes con una fila por viaje: identificador del viaje y del usuario, género, año de nacimiento, fecha y hora de inicio y de fin, y estación de origen y de destino. Un archivo de nomenclatura describe las estaciones (nombre y zona, coordenadas y número de anclajes). La red declara unas 470 estaciones, y 468 tuvieron viajes en septiembre de 2026.

La página de datos abiertos muestra 39,601,506 viajes; la suma de las filas de los 142 archivos es 39,078,984, una diferencia de 522,522 (1.3%). No viene de la lectura, porque cada archivo se lee completo, y no hay una explicación confirmada.

## 3. Planteamiento del problema

**Problema de negocio.** Cuando una estación se queda sin bicicletas o sin espacios, el servicio falla para quien llega a usarla, y la redistribución se vuelve reactiva: se atiende lo que ya está vacío o lleno. Para planearla hace falta saber con anticipación cuántas bicicletas se solicitarán en cada estación y cuándo.

**Problema de Machine Learning.** Predecir, para cada estación s y cada hora t, el número de viajes que se iniciarán en ella. Es un problema supervisado de regresión de conteos. La variable objetivo, `hourly_trip_demand`, se construye contando los viajes cuya estación de origen es s y cuya hora de inicio cae en t.

**Alcance de la Fase 1.** Llegar hasta la tabla de entrenamiento: limpieza, agregación por estación y hora, variables predictoras y su versionado. El entrenamiento y la comparación de modelos corresponden a la Fase 2. Horizonte de predicción: las 24 horas del día siguiente.

## 4. Por qué una solución con ML ayuda

- La demanda depende de la hora, el día de la semana, la temporada y la estación; son cientos de estaciones por miles de horas, y las reglas fijas o los promedios hechos a mano no capturan esas combinaciones a esa escala.
- Un modelo puede combinar variables de calendario, de estación y la historia reciente de cada estación (la demanda de la hora, el día y la semana anteriores) para estimar la demanda esperada.
- Un modelo solo se justifica si mejora a una referencia simple. Por eso el baseline, que repite la demanda de la misma hora de la semana anterior, será la comparación obligatoria en la Fase 2.

## 5. Propuesta de valor

Anticipar dónde y cuándo se necesitará cada bicicleta para que quien opera la redistribución priorice estaciones y horarios antes de que ocurra el faltante, en lugar de reaccionar cuando ya ocurrió. Se benefician la operación (planeación del trabajo de redistribución), los usuarios (mayor disponibilidad de bicicletas y espacios) y la administración del sistema (decisiones apoyadas en datos).

## 6. Requerimientos

| Tipo | Requerimiento |
|---|---|
| Funcional | Estimar los viajes iniciados por estación en cada una de las 24 horas del día siguiente |
| Funcional | Obtener los datos históricos de forma reproducible y construir la tabla de entrenamiento por estación y hora |
| Funcional | Registrar cada versión del dataset (cruda, limpia y agregada) y documentar las modificaciones |
| No funcional | Reproducibilidad y trazabilidad: scripts versionados en GitHub y versiones de datos en MLflow |
| No funcional | Mantenibilidad: código en módulos con la estructura de Cookiecutter Data Science y pruebas automáticas |
| No funcional | Escala: millones de filas; el procesamiento debe caber en 7.8 GB de memoria |
| No funcional | Portabilidad: el pipeline no debe depender de una máquina particular (se verifica en la Fase 3) |

## 7. Machine Learning Canvas

| Bloque | Contenido |
|---|---|
| Proposición de valor | Anticipar cuántos viajes iniciarán en cada estación y hora para decidir dónde y cuándo redistribuir bicicletas. Lo usa quien opera la redistribución; se benefician también los usuarios y la administración. |
| Fuentes de datos | Datos abiertos de MiBici: CSV mensuales de viajes (diciembre de 2014 a septiembre de 2026) y archivo de nomenclatura de estaciones. |
| Recolección de datos | MiBici publica un archivo nuevo por mes; el último mes disponible (septiembre de 2026) está completo, del 1 al 30. |
| Características | Temporales (hora, día de la semana, fin de semana, mes), de estación (identificador, ubicación, capacidad) y de rezago (demanda de la hora, el día y la semana anteriores, y media móvil calculada solo con información previa; con el horizonte de 24 horas se usan los de 24 y 168 horas). |
| Construcción de modelos | Entrenar en la Fase 2 y reentrenar cuando se publique un archivo nuevo, con la periodicidad que defina el equipo. |
| Tareas de Machine Learning | Entrada: estación, hora y rezagos. Salida: número de viajes iniciados. Tipo: regresión de conteos. |
| Decisiones | Usar la demanda esperada para decidir qué estaciones reabastecer o vaciar y en qué horarios. |
| Hacer predicciones | Predicción por lotes, una vez al día: al cierre del día se predice la demanda de cada estación para cada una de las 24 horas del día siguiente. Con este horizonte solo se pueden usar los rezagos de 24 y 168 horas y el calendario: el rezago de 1 hora y la media de las 24 horas anteriores solo existen para la primera hora, y quedan reservados para un horizonte de una hora. |
| Evaluación offline | MAE, RMSE y WAPE con validación temporal (entrenar con el pasado y evaluar en un periodo posterior), contra el baseline de la misma hora de la semana anterior. |
| Monitoreo y evaluación en vivo | Comparar lo predicho con los viajes reales cuando se publique cada archivo mensual y vigilar el drift de las variables de entrada y del error. |

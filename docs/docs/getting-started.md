Primeros pasos
==============

Para dejar el proyecto funcionando desde una instalación limpia (Python 3.10 o superior y unos 8 GB de memoria):

1. Crear y activar el entorno, e instalar las dependencias:

        python3 -m venv .venv && source .venv/bin/activate
        python -m pip install -r requirements.txt

2. Comprobar que todo funciona:

        python -m pytest -q

3. Obtener y limpiar los datos (descarga unos 3 GB de archivos mensuales de MiBici a `data/raw`):

        python -m mibici_demand.cli descargar
        python -m mibici_demand.cli procesar

4. Construir la tabla de entrenamiento y registrar la versión de los datos en MLflow:

        python -m mibici_demand.cli construir
        python -m mibici_demand.cli registrar --version v1

5. Ver los registros en la interfaz de MLflow:

        python -m mlflow ui --backend-store-uri sqlite:///mlflow.db

Para comparar una ventana de años distinta se usa `--desde 2022-01 --etiqueta desde2022` en `construir` y en `registrar`.

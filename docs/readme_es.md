# Airedellín 🌿 (En Progreso 🏗️🚧)

<center>
<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/logo.png" alt="Descripción" style="width: 50%; height: 50%;">  
</div>
</center>

**Airedellín** es un repositorio iniciado para promover la ciencia ciudadana en Medellín, con el objetivo de abordar el problema de la calidad del aire y las PM2.5. La plataforma se basa en proyectos de grupos ciudadanos como Canairio y Unloquer, con un enfoque en visualizar y analizar datos de calidad del aire para mejorar la salud pública.

## Resumen del Proyecto

Medellín experimenta niveles elevados de PM2.5, especialmente durante septiembre y abril-mayo. Las partículas PM2.5, que tienen un diámetro de 2.5 micrómetros, provienen de combustibles no limpios y diversas fuentes como la cocina, las actividades industriales y los vehículos. Estas partículas finas pueden penetrar profundamente en los pulmones y el torrente sanguíneo, planteando graves riesgos para la salud como enfermedades cardíacas y pulmonares, accidentes cerebrovasculares y cáncer. Los vientos del Sahara también contribuyen a estos aumentos.

Airedellín utiliza tecnologías de vanguardia para enfrentar este desafío y visualizar los datos de calidad del aire, utilizando:

- **Python** & **FastAPI**: Para el desarrollo del backend y la API.
- **JavaScript**: Para las interacciones en el frontend.
- **Deck.gl** & **MapLibre**: Para visualizaciones de mapas bellas y responsivas.
- **Bootstrap**: Para una interfaz de usuario elegante y moderna.
- **InfluxDB**: Para el almacenamiento y consulta eficiente de datos.
- **CanAirIO**: Proveedor de datos en tiempo real sobre la calidad del aire en Medellín.
- **Otras librerías**: como Tensorflow, Xgboost, Scikit-learn, Statsmodels

La plataforma incluye modelos de aprendizaje automático para analizar y predecir patrones de calidad del aire basados en datos históricos.

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/2.png)

### Modelos y Algoritmos de Aprendizaje Automático

Los siguientes modelos se utilizan para predecir y analizar las tendencias de calidad del aire:

- **Regresión Lineal**: Predice la relación entre la calidad del aire y diversos factores.
- **ARIMA & SARIMA**: Modelos de series temporales, adecuados para datos con patrones temporales y estacionales.
- **Random Forest**: Efectivo para problemas complejos con múltiples variables.
- **Lasso**: Regresión lineal regularizada, útil para reducir la complejidad del modelo.
- **XGBoost**: Potente algoritmo basado en árboles de decisión utilizando boosting.
- **Suavizado Exponencial**: Para suavizar cambios en datos de series temporales.
- **LSTM**: Una red neuronal recurrente ideal para datos basados en secuencias como la calidad del aire.
- **Suavizado Exponencial**: Para cambios suaves a lo largo del tiempo.

Algunos de estos modelos están diseñados con fines educativos.

## Características

- **Visualización en Mapa**: Airedellín incluye un mapa interactivo en tiempo real que muestra los datos de calidad del aire de varios sensores en la ciudad. Los sensores están codificados por colores según los datos, y al hacer clic en un sensor se muestran ventanas emergentes con información detallada.
- **Mapa de Calor**: Los usuarios pueden alternar un mapa de calor para visualizar la intensidad de las concentraciones de PM2.5 en la ciudad.
- **Relieve 3D**: Añade una capa visual adicional para hacer el mapa más informativo.
- **Predicciones**: Predice los niveles de calidad del aire utilizando modelos de aprendizaje automático.

### Páginas

- `/`: Página de inicio del visualizador de datos.
- `/sensor{sensor_name}`: Muestra los datos del sensor y permite filtrar por rango de fechas.
- `/sensor{sensor_name}/statistics`: Muestra estadísticas del sensor como la media, la varianza y la desviación estándar.
- `/sensor{sensor_name}/predictions`: Permite a los usuarios seleccionar algoritmos de predicción y ver el resultado.
- `/add_donation`: Página para donar y apoyar los sensores.
- `/index`: Página principal del sitio para presentar el proyecto.

## Capturas de Pantalla 🎑

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/1.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/3_new.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/4_new.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/5.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/6.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/7.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/8.png)


---

## Cómo Ejecutar 🏃‍♀️

Sigue estos pasos para comenzar con Airedellín:

1. **Clona el repositorio**:

    ```bash
    git clone https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin.git
    cd airedellin
    ```

2. **Configura tu entorno**:

    - Asegúrate de tener Python 3.9+
    - Crea un entorno virtual:
    
        ```bash
        python -m venv env
        source env/bin/activate  # En Windows: `env\Scripts\activate`
        ```

    - Instala las dependencias requeridas:
    
        ```bash
        pip install -r requirements.txt
        ```

    - Añade tu token de Maptiler en `data/token.txt`.

3. **Inicia la aplicación**:

    ```bash
    uvicorn airedellin:app --host 0.0.0.0 --port 9600
    ```

4. **Accede a la interfaz web**:

    Visita `http://localhost:9600` para comenzar a explorar los datos de calidad del aire de Medellín.

---

## Contribuciones 🤝

Airedellín es un proyecto de código abierto impulsado por la comunidad. Agradecemos todas las formas de contribuciones, incluyendo código, documentación, diseño o retroalimentación. No dudes en enviar problemas o solicitudes de extracción en GitHub.

---

### Agradecimientos 💚

Agradecimientos especiales a **Hackerspace Unloquer** y **CanAirIO** por inspirar y apoyar este proyecto. ¡Su contribución a los esfuerzos de calidad del aire en Medellín es invaluable!

Únete a nosotros para mejorar la calidad del aire en Medellín para todos. 🚀🌱

---

### Notas Importantes

- **Datos de París**: Los datos utilizados para los sensores de París no son reales; son para fines de prueba.


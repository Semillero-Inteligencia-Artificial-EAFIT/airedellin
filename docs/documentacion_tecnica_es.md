
# Airedellín 🌿 (en progreso 🏗️🚧)

Repositorio como iniciativa para incentivar la ciencia ciudadana en Medellín para abordar el problema de calidad de aire y PM2.5 en la ciudad de Medellín. Se basa en otros trabajos de grupos ciudadanos como Canairio y Unloquer. En Colombia, en Medellín sucede un fenómeno en septiembre y entre abril-mayo, donde se eleva la concentración de PM2.5. El PM2.5 es una partícula fina de 2.5 micrómetros de radio, estas provienen de combustibles no limpios, como la cocina (cuando se quema la arepa) o la calefacción, la quema de basuras y residuos agrícolas, las actividades industriales, el transporte y el polvo arrastrado por el viento, entre otras fuentes. Las partículas PM2.5 penetran profundamente en los pulmones y el torrente sanguíneo, lo que aumenta el riesgo de enfermedades cardíacas, pulmonares, derrames cerebrales y cáncer. Estas partículas pueden ser emitidas directamente o formarse en la atmósfera a partir de contaminantes emitidos, como el amoníaco y los compuestos orgánicos volátiles.

La materia particulada incluye sustancias químicas orgánicas (como polen), polvo, hollín y metales.

En Medellín, las causas del aumento de PM2.5 son incendios forestales, vehículos, fábricas y, más importante, los vientos del Sahara.

Nuestra plataforma trata de ser lo más regionalista posible, utilizando proyectos colombianos como FastAPI y herramientas similares. Nos inspiramos en el visualizador de datos.

FastAPI es una herramienta eficiente, permite asincronía, un modo de inicialización que necesitamos para crear un hilo que se ejecute cada 30 minutos para obtener los datos de los sensores.

Cuenta con un mapa hecho con MapLibre y DeckGL, la librería que utiliza Uber (el de los carros, no Uber de Unloquer).

En el mapa hay unos sensores en París que son para calibrar colores y verificar que todo funcione.

Usamos MapLibre con DeckGL porque Pydeck (la versión de DeckGL para Python) no permite hacer pop-ups clicables, sino pop-ups imposibles de oprimir 

video: ![](https://private-user-images.githubusercontent.com/11672957/364573233-db9f9d25-aaa7-47b8-a8d7-e9ed990a5759.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU4NDg4MTcsIm5iZiI6MTcyNTg0ODUxNywicGF0aCI6Ii8xMTY3Mjk1Ny8zNjQ1NzMyMzMtZGI5ZjlkMjUtYWFhNy00N2I4LWE4ZDctZTllZDk5MGE1NzU5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTA5VDAyMjE1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBjYTY2Nzk3YjY0M2ZiNDZlNTBhYjczYzFiYmM3NDVhOGY4ZjIwNDRiNWEwZWQzZWRjM2YyMWEzZjE1MzE5NTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1Gwypn8rVxo8dKgpwevuisWAznlhV6po2Pos-O9oO3s) 

y porque nos permite crear animaciones, elementos en 3D y más.

Necesitamos poder hacer clic en los puntos que son sensores de calidad de aire de Canairio para obtener información de ellos. El pop-up contiene dos hipervínculos, uno al gráfico de Canairio y otro a nuestra página con una sección aparte para el sensor clicado. En esta sección podemos ver un historial del tiempo solo de PM2.5 y unos botones para mejorar la visualización: zoom in, zoom out, y seleccionar datos de fechas más antiguas.

Podemos desactivar el heatmap o activarlo con un botón.

También tiene un relieve 3D atractivo.

Hay 3 secciones especiales para los sensores en la página del sensor:
- Donaciones
- Estadísticas
- Machine learning y predicciones

Las donaciones se pensaron como un incentivo, propuesto por el hackerspace de Unloquer. Se creó un mecanismo universal basado en la confianza que permite que un usuario añada a un sensor datos para donar más un QR. El problema es la confianza, por lo que queremos usar Nostr para no tener la responsabilidad de los sensores.

Cuenta con algoritmos de machine learning, ya que en el semillero de machine learning hicimos un ejercicio para implementar algoritmos para series de tiempo. Muchos de estos algoritmos se pueden mejorar y no son lo mejor para este tipo de tareas.

Usamos estos algoritmos y modelos:

- **OriginalData**: Se refiere a los datos crudos o sin procesar que se utilizan para entrenar los modelos y hacer predicciones.
- **Linear Regression**: Modelo estadístico simple que predice la relación entre una variable dependiente y una o más variables independientes.
- **ARIMA**: Modelo de promedio móvil e integración autorregresiva, útil para datos de series temporales.
- **Random Forest**: Algoritmo de aprendizaje automático basado en múltiples árboles de decisión, adecuado para problemas complejos.
- **SARIMA**: Extensión del modelo ARIMA que incluye estacionalidad.
- **Lasso**: Modelo de regresión lineal con regularización para evitar el sobreajuste.
- **XGBoost**: Algoritmo de árboles de decisión con boosting, eficiente para predicciones complejas.
- **Exponential Smoothing**: Método de suavizado exponencial para predicciones de series temporales.
- **LSTM**: Red neuronal recurrente que maneja secuencias de datos y es ideal para series temporales.

Tratamos de usar Prophet de Facebook, pero sigue siendo Facebook, no funciona y está desactualizado.

**Estadísticas**

Creamos una página para observar las estadísticas de cada sensor:
- Suma
- Media
- Varianza
- Desviación estándar
- Máximo
- Mínimo
- Moda
- Curtosis
- Asimetría
- Coeficiente de variación (CV)
- Conteo de valores nulos

Con gráficos no solo del PM2.5, sino también de PM1 y PM10. Gráficos con Z-score para observar los cambios respecto al promedio y un gráfico de barras con los valores más comunes.

### Rutas de la aplicación:
- **/**: Visualizador.
- **/sensor{sensor_name}**: Muestra datos del sensor (GET) y recibe un rango de tiempo para mostrar datos en ese rango (POST).
- **/sensor{sensor_name}/statistics**: Muestra estadísticas del sensor.
- **/sensor{sensor_name}/predictions**: Permite seleccionar algoritmos de predicción y muestra resultados.
- **/index**: Página de inicio.
- **/add_donation**: Añadir una donación.

### Archivos importantes:
- **tools.py**: Funciones útiles para manejar datos y archivos.
- **pred.py**: Funciones para los algoritmos de machine learning.
- **dummy_donations.py**: Manejo de donaciones, aunque queremos migrar a Nostr.
- **datatool.py**: Conexión con la base de datos de Canairio y manipulación de datos.
- **const.py**: Guardado de datos importantes.

En la carpeta **core/static** guardamos los QR dummy, el CSS de las páginas y un poco de JavaScript. No podemos migrar el JavaScript de los archivos HTML externamente porque se daña la gráfica debido a cómo Jinja carga el código de Python a variables de JavaScript.

En la carpeta **test/** se realizan pruebas, aunque está algo desordenada.

Nos estamos enfocando en que la construcción de esta página sea útil para practicar para la hackatón de la NASA. No podemos usar este proyecto porque, si queremos ganar, tenemos que usar sus datos.


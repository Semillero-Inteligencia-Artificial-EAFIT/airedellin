# Airedellín 🌿 (en progresso 🏗️🚧)

repositorio como iniciativa para incentivar la ciencia ciudadana en medellin para aboradar el problema de calidad de aire y pm2.5 en la ciudad de medellin , en base a otros trabajos de grupos ciudadanos como canairio y unloquer, en colombia en medellin sucede un fenomeno en septiembre y entre abril-mayo que se eleva la concentracion de pm2.5, el pm2.5 es una particula fina de 2.5 micrometros de radio , estas vienen de combustibles no limpios 

 la cocina (cuando se quema la arepa) o la calefacción, la quema de basuras y residuos agrícolas, las actividades industriales, el transporte y el polvo arrastrado por el viento, entre otras fuentes. Las partículas PM2,5 penetran profundamente en los pulmones y el torrente sanguíneo, lo que aumenta el riesgo de morir por enfermedades cardiacas y pulmonares, derrames cerebrales y cáncer. Estas partículas pueden ser emitidas directamente o formarse en la atmósfera a partir de distintos contaminantes emitidos, como el amoníaco y los compuestos orgánicos volátiles.

 La materia particulada incluye sustancias químicas orgánicas (como polem), polvo, hollín y metales

en medellin las causas del aumento de pm2.5 son incendios forestales, vehiculos, fabricas y mas importante de los vientos del sahara




nuestra plataforma trata de ser lo mas regionalista posible usando proyectos colombianos como fast api y herramientas como fastapi, nos inspiramos en el visualizador de datos

fastapi es una herramienta eficiente,permite async, un modo de inicializacion que nesesitamos para obtener al inicio para crear un hilo que se ejecute cada 30 minutos para obtener los datos de los sensores


cuenta con un mapa hecho con maplibre y deckgl la libreria que usa uber(el de los carros, no uber de unloquer)

en el mapa hay unos sensores en paris que son para calibrar colores y verificar que todo funcione

usamos maplibre con deckgl por que pydeck (la version de deck gl de python) no permite hacer pop ups cliquiables, permite hacer pop ups imposibles de oprimir (video:https://private-user-images.githubusercontent.com/11672957/364573233-db9f9d25-aaa7-47b8-a8d7-e9ed990a5759.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU4NDg4MTcsIm5iZiI6MTcyNTg0ODUxNywicGF0aCI6Ii8xMTY3Mjk1Ny8zNjQ1NzMyMzMtZGI5ZjlkMjUtYWFhNy00N2I4LWE4ZDctZTllZDk5MGE1NzU5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTA5VDAyMjE1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBjYTY2Nzk3YjY0M2ZiNDZlNTBhYjczYzFiYmM3NDVhOGY4ZjIwNDRiNWEwZWQzZWRjM2YyMWEzZjE1MzE5NTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1Gwypn8rVxo8dKgpwevuisWAznlhV6po2Pos-O9oO3s) y por que nos permite crear animaciones , cositas en 3d y mas

nesesitamos poder cliquear en los puntos que son sensores de calidad de aire canairio para obtener informacion de ellos, el popup contiene 2 hyperviculos uno al grafa de canairio y otro a nuestra pagina con una seccion aparta para el sensor que se cliquehe, en esta seccion podemos ver un historial del tiempo solo de pm2.5 y unos botones para mejorar la visualizacion ,zoom in y zoom out, y selecionar datos de fechas mas antiguas

podemos desactivar el heatmap o activarlo con un boton

tambien tiene un facy relieve 3d que hace la pagina mas atractiva

hay 3 secciones especiales para los sensores en la pagina del sensor

donaciones,
estadisticas,
y machine learnig y predicciones


las donaciones se pensaron como un incetivo y por que las propusieron del hackerspace de unloquer, se dejo hecho un mecanismo universal basado en la confiasa que perimte que un usuario añada a un sensor datos para donarle mas un QR, el problema de esto es la confianza, por eso queremos usar nostr para no tener la responsabilidad de los sensores 

cuenta con algoritmos de machine learning por que en el semillero de machine learning hicimos un ejercio para implementar algoritmos para series de tiempo, muchos de estos algoritmos se pueden mejorar y no son lo mejor para este tipo de tareas

usamos estos algorimos y modelos

OriginalData: Se refiere a los datos crudos o sin procesar que se utilizan para entrenar los modelos y hacer predicciones. No es un algoritmo, pero es la base para aplicar los modelos.

Linear Regression: Es un modelo estadístico simple que trata de predecir la relación entre una variable dependiente (en este caso, la calidad del aire) y una o más variables independientes. Útil cuando la relación entre las variables es lineal.

ARIMA: Modelo de promedio móvil e integración autoregresiva, usado para datos de series temporales. Puede ser útil para predecir la calidad del aire cuando hay patrones temporales.

Random Forest: Un algoritmo de aprendizaje automático basado en múltiples árboles de decisión. Es adecuado para problemas complejos con muchas variables, por lo que podría ser útil para predecir calidad de aire.

SARIMA: Es una extensión del modelo ARIMA que incluye la estacionalidad, lo que lo hace útil para series temporales con patrones estacionales (como los cambios estacionales en la calidad del aire).

Lasso: Un modelo de regresión lineal que incluye regularización para evitar el sobreajuste. Puede ser útil si hay muchas variables relacionadas con la calidad del aire, ya que reduce la complejidad del modelo.

XGBoost: Un algoritmo de árboles de decisión que utiliza boosting, lo que lo hace potente y eficiente para predicciones complejas, incluidas las de calidad del aire.

Exponential Smoothing: Este método de suavizado exponencial se usa para hacer predicciones a partir de datos de series temporales. Es útil cuando los datos cambian suavemente a lo largo del tiempo.

LSTM: Una red neuronal recurrente que maneja secuencias de datos y es ideal para predicciones basadas en series temporales, como la calidad del aire, debido a su capacidad para recordar dependencias a largo plazo.

tratamos de usar prophet de facebook pero sigue siendo facebook , no funciona y esta desactualizado por eso sigue siendo facebook

las estadisticas

hicimos una pagina para observar las estadisticas cada sensor, 

Suma
Media
Varianza
Desviación estándar
Máximo
Mínimo
Moda
Curtosis
Asimetría
Coeficiente de variación (CV)
Conteo de valores nulos

con graficas no solo del pm2.5 tambien con el pm1, y pm10  

graficos con Z-score para observar como cambian respecto al promedio y una grafica de barras con los valores mas comunes

hay paginas como

/ donde esta el visualizador

/sensor{sensor_name}
Página get: Muestra datos del sensor.
Página post: Recibe un rango de tiempo desde un formulario para mostrar datos del sensor en ese rango.

/sensor{sensor_name}/statistics
Página: Muestra estadísticas del sensor.


/sensor{sensor_name}/predictions


Página get: Muestra la página para seleccionar algoritmos de predicción.
Página post: Procesa el algoritmo seleccionado y muestra el resultado de la predicción.

/index

Página: Página de inicio (landing page).

/add_donation
Página: Muestra la página para añadir una donación.

en la carpeta tools hay archivos como 

tools.py
pred.py
dummy_donations.py
datatool.py
const.py


tools.py

algunas funciones utiles para manejar datos y archivos

pred.py

archivo donde tenemos funciones para los algoritmos de machine learning

dummy_donations.py

archivo para manejar las donaciones, como nos queremos mover a nostr, lo dejamos y es un producto que quien sabe que luego usemos

datatool.py

hay un codigo que nos facilita la conexion con la base de datos de canairio y herramientas para manipular los datos

const.py

deberia usarla para guardar datos imporatnes


hay una carpeta core/static donde se guarda los qr dummys, el css de las paginas y un poco de javascript, no podemos migrar el javascript de los archivos de html externamente por que se nos daña la grafica por los templates de jinga cargan el codigo que va desde python a variables de javascript

tambien hay una carpeta test/ son de se hacen pruebas, se deja un desorden

y nos estamos enfocado en que la construccion de esta pagina sea util para practicar para la hackaton de la nasa, no podemos usar este proyecto por que si queremos ganar tenemos que usar los datos de ellos
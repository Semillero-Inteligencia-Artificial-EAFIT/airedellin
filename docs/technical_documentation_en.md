# Airedellín 🌿 (In Progress 🏗️🚧)

A repository initiated to promote citizen science in Medellín, addressing the issue of air quality and PM2.5 in the city. It builds upon other projects from citizen groups like Canairio and Unloquer. In Medellín, there is a phenomenon in September and between April-May where the concentration of PM2.5 increases. PM2.5 is a fine particle with a radius of 2.5 micrometers, coming from unclean fuels.

These particles originate from activities such as cooking (e.g., burning arepas), heating, burning of waste and agricultural residues, industrial activities, transportation, and wind-driven dust. PM2.5 particles penetrate deep into the lungs and bloodstream, increasing the risk of death from heart and lung diseases, strokes, and cancer. These particles can be emitted directly or formed in the atmosphere from various pollutants like ammonia and volatile organic compounds.

Particulate matter includes organic chemicals (such as pollen), dust, soot, and metals.

In Medellín, the causes of PM2.5 increases are wildfires, vehicles, factories, and most importantly, winds from the Sahara.

Our platform aims to be as regionally focused as possible, using Colombian projects like FastAPI and tools such as FastAPI. We were inspired by the data visualizer.

FastAPI is an efficient tool, allowing async operation and initialization mode, which we need to start a thread that runs every 30 minutes to gather data from sensors.

The platform includes a map built with MapLibre and Deck.gl, the library used by Uber (the ride-hailing service, not Unloquer’s Uber).

On the map, there are sensors in Paris that are used to calibrate colors and ensure everything works.

We use MapLibre with Deck.gl because Pydeck (the Python version of Deck.gl) does not allow clickable pop-ups—it only permits unclickable pop-ups video: ![](https://private-user-images.githubusercontent.com/11672957/364573233-db9f9d25-aaa7-47b8-a8d7-e9ed990a5759.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjU4NDg4MTcsIm5iZiI6MTcyNTg0ODUxNywicGF0aCI6Ii8xMTY3Mjk1Ny8zNjQ1NzMyMzMtZGI5ZjlkMjUtYWFhNy00N2I4LWE4ZDctZTllZDk5MGE1NzU5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA5MDklMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwOTA5VDAyMjE1N1omWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTBjYTY2Nzk3YjY0M2ZiNDZlNTBhYjczYzFiYmM3NDVhOGY4ZjIwNDRiNWEwZWQzZWRjM2YyMWEzZjE1MzE5NTMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.1Gwypn8rVxo8dKgpwevuisWAznlhV6po2Pos-O9oO3s)  and because it allows creating animations, 3D elements, and more.

We need clickable points that are air quality sensors from Canairio to get information about them. The pop-up contains two hyperlinks—one to Canairio's graph and another to our webpage, with a dedicated section for the clicked sensor. In this section, we can view a historical timeline of PM2.5 data, along with buttons to improve visualization, zoom in and out, and select older date ranges.

The heatmap can be activated or deactivated with a button.

There’s also a 3D relief feature that makes the page more visually appealing.

There are three special sections for sensors on the sensor page:
- Donations,
- Statistics,
- Machine learning and predictions.

Donations were proposed as an incentive and suggested by Unloquer’s hackerspace. We designed a universal mechanism based on trust that allows users to add data to a sensor for donation, along with a QR code. The challenge is trust, which is why we want to use Nostr to avoid being responsible for sensors.

The platform includes machine learning algorithms because, in the machine learning research group, we conducted an exercise to implement time series algorithms. Many of these algorithms can be improved, and they are not optimal for these tasks.

We use the following algorithms and models:

- **OriginalData**: Refers to the raw data used for training models and making predictions. It is the foundation for applying the models, though not an algorithm itself.
- **Linear Regression**: A simple statistical model to predict the relationship between a dependent variable (in this case, air quality) and one or more independent variables. It is useful when the relationship between variables is linear.
- **ARIMA**: An autoregressive moving average model used for time series data. It can be useful for predicting air quality when there are temporal patterns.
- **Random Forest**: A machine learning algorithm based on multiple decision trees. Suitable for complex problems with many variables, making it potentially useful for air quality prediction.
- **SARIMA**: An extension of ARIMA that includes seasonality, making it useful for time series with seasonal patterns (like seasonal changes in air quality).
- **Lasso**: A linear regression model with regularization to prevent overfitting. It can be useful when there are many variables related to air quality, as it reduces model complexity.
- **XGBoost**: A decision tree-based algorithm that uses boosting, making it powerful and efficient for complex predictions, including air quality.
- **Exponential Smoothing**: This method is used for forecasting based on time series data. It is helpful when data changes smoothly over time.
- **LSTM**: A recurrent neural network that handles data sequences, ideal for time series predictions like air quality due to its ability to remember long-term dependencies.

We tried using Facebook’s Prophet but it’s outdated and unreliable—still Facebook, after all.

### Statistics

We created a page to observe statistics for each sensor:

- Sum,
- Mean,
- Variance,
- Standard deviation,
- Maximum,
- Minimum,
- Mode,
- Kurtosis,
- Skewness,
- Coefficient of variation (CV),
- Count of null values.

With graphs not only for PM2.5 but also for PM1 and PM10.

Z-score graphs help observe how values deviate from the average, and a bar chart shows the most common values.

Pages include:

- `/`: Where the data visualizer is located.
- `/sensor{sensor_name}`: 
    - GET page: Shows sensor data.
    - POST page: Receives a time range from a form to display data from that sensor within the range.
- `/sensor{sensor_name}/statistics`: Shows sensor statistics.
- `/sensor{sensor_name}/predictions`: 
    - GET page: Displays a page to select prediction algorithms.
    - POST page: Processes the selected algorithm and shows the prediction result.
- `/index`: Landing page.
- `/add_donation`: Page for adding a donation.

The `tools` folder contains files like:

- **tools.py**: Useful functions for handling data and files.
- **pred.py**: Contains functions for machine learning algorithms.
- **dummy_donations.py**: Handles donations. Since we want to transition to Nostr, this is left as a potential future tool.
- **datatool.py**: Code to facilitate connection with Canairio’s database and tools for manipulating data.
- **const.py**: Meant to store important data.

The `core/static` folder holds dummy QR codes, CSS files for the pages, and some JavaScript. We cannot migrate the JavaScript from external HTML files because the Jinja templates load the code from Python into JavaScript variables.

There is also a `test/` folder where tests are conducted—though it’s a bit disorganized.

We are focusing on building this platform as practice for the NASA hackathon. However, we can't use this project for the competition because, to win, we need to use their data.
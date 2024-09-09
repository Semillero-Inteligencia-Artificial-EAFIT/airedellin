# Airedellín 🌿 (In Progress 🏗️🚧)

<center>
<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/logo.png" alt="Description" style="width: 50%; height: 50%;">  
</div>
</center>

**Translations in**
[Español](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/readme_es.md)
[Deutsch](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/readme_de.md) 
[Русский](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/readme_ru.md)

**Airedellín** is a repository initiated to promote citizen science in Medellín, aiming to address the issue of air quality and PM2.5. The platform builds upon projects from citizen groups like Canairio and Unloquer, with a focus on visualizing and analyzing air quality data to improve public health.

## Project Overview

Medellín experiences heightened PM2.5 levels, particularly during September and April-May. PM2.5 particles, which are 2.5 micrometers in diameter, come from unclean fuels and various sources like cooking, industrial activities, and vehicles. These fine particles can penetrate deep into the lungs and bloodstream, posing severe health risks such as heart and lung diseases, strokes, and cancer. Winds from the Sahara also contribute to these increases.

Airedellín leverages cutting-edge technologies to tackle this challenge and visualize air quality data, using:

- **Python** & **FastAPI**: For backend and API development.
- **JavaScript**: For frontend interactions.
- **Deck.gl** & **MapLibre**: For beautiful, responsive map visualizations.
- **Bootstrap**: For a sleek and modern UI.
- **InfluxDB**: For efficient data storage and querying.
- **CanAirIO**: Real-time air quality data provider for Medellín.
- **Other libraris**: like Tensorflow, Xgboost ,Scikit-learn, Statsmodels



The platform includes machine learning models to analyze and predict air quality patterns based on historical data.

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/2.png)

### Machine Learning Models and Algorithms

The following models are used to predict and analyze air quality trends:

- **Linear Regression**: Predicts the relationship between air quality and various factors.
- **ARIMA & SARIMA**: Time series models, suitable for data with temporal and seasonal patterns.
- **Random Forest**: Effective for complex problems with multiple variables.
- **Lasso**: Regularized linear regression, helpful for reducing model complexity.
- **XGBoost**: Powerful decision tree-based algorithm using boosting .
- **Exponential Smoothing**: For smooth time series data changes.
- **LSTM**: A recurrent neural network ideal for sequence-based data like air quality.
- **Exponential Smoothing**: For smooth changes over time.

some of this modeles are made for learning propuses

## Features

- **Map Visualization**: Airedellín includes a real-time, interactive map displaying air quality data from various sensors across the city. Sensors are color-coded based on the data, and clicking on a sensor displays pop-ups with detailed information.
- **Heatmap**: Users can toggle a heatmap to visualize the intensity of PM2.5 concentrations across the city.
- **3D Relief**: Adds an extra visual layer to make the map more informative.
- **Predictions**: Predict air quality levels using machine learning models.

### Pages

- `/`: Data visualizer home.
- `/sensor{sensor_name}`: Displays sensor data and allows filtering by date range.
- `/sensor{sensor_name}/statistics`: Shows sensor statistics like mean, variance, and standard deviation.
- `/sensor{sensor_name}/predictions`: Lets users select prediction algorithms and view the result.
- `/add_donation`: Page to donate and support sensors.
- `/index`: index website for present the project


## Screenshots 🎑

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/1.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/3_new.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/4_new.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/5.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/6.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/7.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/8.png)


---

## How to Run 🏃‍♀️

Follow these steps to get started with Airedellín:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin.git
    cd airedellin
    ```

2. **Set up your environment**:

    - Make sure you have Python 3.9+
    - Create a virtual environment:
    
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows: `env\Scripts\activate`
        ```

    - Install the required dependencies:
    
        ```bash
        pip install -r requirements.txt
        ```

    - Add your Maptiler token in `data/token.txt`.

3. **Start the application**:

    ```bash
    uvicorn airedellin:app --host 0.0.0.0 --port 9600
    ```

4. **Access the web interface**:

    Visit `http://localhost:9600` to start exploring Medellín’s air quality data.

---

## Contributing 🤝

Airedellín is an open-source, community-driven project. We appreciate all forms of contributions, including code, documentation, design, or feedback. Feel free to submit issues or pull requests on GitHub.

---

### Acknowledgements 💚

Special thanks to **Hackerspace Unloquer** and **CanAirIO** for inspiring and supporting this project. Your contribution to Medellín’s air quality efforts is invaluable!

Join us in improving Medellín’s air quality for everyone. 🚀🌱

---

### Important Notes

- **Paris Data**: The data used for Paris sensors is not real; it's for testing purposes.
- **Techincal Notes**: [Here](https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/blob/main/docs/technical_documentation_en.md)
### We want to

- [X] Heatmap
- [X] Plot sensors in map
- [X] Sensors with real pm2.5 values
- [X] Predict with some algorithms the value of pm2.5
- [X] Create dashboard for each sensor
- [X] Donations dummy system
- [X] Stadistical panel of pm2.5
- [ ] Hexgon map like in [https://sensor.community/es/](https://sensor.community/es/)
- [ ] change the time when the sensor take data for better predictions , predict not from 30 sec to 30 sec. for 1 day to another day 
- [ ] Real location of sensors
- [ ] Waze for pm2.5
- [ ] Web with layers for predict pm2.5 in a anothermap

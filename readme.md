# Airedellín 🌿 (In progres 🏗️🚧)

<center>
<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/logo.png" alt="Description" style="width: 50%; height: 50%;">  
</div>
</center>

Welcome to **Airedellin** , a PM2.5 visualizer designed to monitor and improve the air quality in Medellín! 🌆✨

**This project is currently under development  🚧**, and we would love for you to join us! Our team, **MLEAFIT** (Machine Learning club from university EAFIT), is dedicated to applying ML techniques to tackle air quality challenges. We are preparing to participate in the **NASA Space Apps Challenge** with a mission to help from Medellín breathe cleaner air to the word.

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/2.png)

## Project Overview

Airedellín leverages several cutting-edge technologies:

- **Python** & **FastAPI**: For backend and API development.
- **JavaScript**: For frontend interactions.
- **Deck.gl** & **MapLibre**: For beautiful and responsive map visualizations.
- **Bootstrap**: For a sleek and modern UI.
- **InfluxDB**: For efficient querying.
- **CanAirIO**: Our data source, providing real-time air quality data from Medellín.

## How to Run 🏃‍♀️

To get started with Airedellín, follow these steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin.git
    cd airedellin
    ```

2. **Set up your environment**:

    - Make sure you have Python 3.9+ 
    - Create a virtual environment for the Python dependencies:
    
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows, use `env\Scripts\activate`
        ```

    - Install the required Python packages:
    
        ```bash
        pip install -r requirements.txt
        ```
    - Rember add a file token and the token of maptiler in data/token.txt

3. **Start the application**:

    Run the following command to launch the FastAPI server:

    ```bash
    uvicorn airedellin:app --host 0.0.0.0 --port 9600
    ```

4. **Access the web interface**:

    Open your browser and go to `http://localhost:9600` to start exploring the air quality data of Medellín! 🌍✨

## Screenshots 🎑

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/1.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/3.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/4.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/5.png)


## Contributing 🤝

Airedellín is a community-driven project. We appreciate contributions in all forms—whether it's code, documentation, design, or feedback! If you would like to help, feel free to open an issue or submit a pull request.

### Acknowledgements 💚

Special thanks to **Hackerspace Unloquer** and **CanAirIO** for inspiring and motivating us to create this project. Your support is invaluable!

---

Join us in making Medellín's air quality better for everyone! Together, we can make a difference. 🚀🌱

### Important Notes

 - paris data is not real is for cheaking everything is okey



## TODO


- [X] Heatmap
- [X] Plot sensors in map
- [X] Sensors with real pm2.5 values
- [X] Predict with some algorithms the value of pm2.5
- [X] Create dashboard for each sensor
- [ ] Donations dummy system
- [ ] Stadistical panel of pm2.5 
- [ ] Real location of sensors
- [ ] Waze for pm2.5
- [ ] Web with layers for predict pm2.5 in a anothermap
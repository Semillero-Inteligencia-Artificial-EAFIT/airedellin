# Airedellín 🌿 (In Bearbeitung 🏗️🚧)(nicht überarbeitete Übersetzung)

<center>
<div style="text-align: center;">
<img src="https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/logo.png" alt="Beschreibung" style="width: 50%; height: 50%;">  
</div>
</center>

**Airedellín** ist ein Repository, das ins Leben gerufen wurde, um die Bürgerwissenschaft in Medellín zu fördern, mit dem Ziel, das Problem der Luftqualität und PM2.5 anzugehen. Die Plattform basiert auf Projekten von Bürgergruppen wie Canairio und Unloquer und konzentriert sich darauf, Daten zur Luftqualität zu visualisieren und zu analysieren, um die öffentliche Gesundheit zu verbessern.

## Projektübersicht

In Medellín sind die PM2.5-Werte besonders im September und April-Mai erhöht. PM2.5-Partikel mit einem Durchmesser von 2,5 Mikrometern stammen von unreinen Brennstoffen und verschiedenen Quellen wie Kochen, Industrieaktivitäten und Fahrzeugen. Diese feinen Partikel können tief in die Lunge und den Blutkreislauf eindringen und schwerwiegende Gesundheitsrisiken wie Herz- und Lungenkrankheiten, Schlaganfälle und Krebs darstellen. Auch Winde aus der Sahara tragen zu diesen Erhöhungen bei.

Airedellín nutzt moderne Technologien, um diese Herausforderung zu bewältigen und Daten zur Luftqualität zu visualisieren, indem es verwendet:

- **Python** & **FastAPI**: Für Backend- und API-Entwicklung.
- **JavaScript**: Für Frontend-Interaktionen.
- **Deck.gl** & **MapLibre**: Für schöne, reaktionsschnelle Kartenvisualisierungen.
- **Bootstrap**: Für ein elegantes und modernes UI.
- **InfluxDB**: Für effiziente Datenspeicherung und -abfrage.
- **CanAirIO**: Echtzeit-Luftqualitätsdatenanbieter für Medellín.
- **Andere Bibliotheken**: wie Tensorflow, Xgboost, Scikit-learn, Statsmodels

Die Plattform umfasst Modelle des maschinellen Lernens zur Analyse und Vorhersage von Luftqualitätsmustern auf Basis historischer Daten.

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/2.png)

### Modelle und Algorithmen des maschinellen Lernens

Die folgenden Modelle werden verwendet, um Luftqualitätstrends vorherzusagen und zu analysieren:

- **Lineare Regression**: Vorhersage der Beziehung zwischen Luftqualität und verschiedenen Faktoren.
- **ARIMA & SARIMA**: Zeitreihenmodelle, geeignet für Daten mit zeitlichen und saisonalen Mustern.
- **Random Forest**: Effektiv für komplexe Probleme mit mehreren Variablen.
- **Lasso**: Regularisierte lineare Regression, hilfreich zur Reduzierung der Modellkomplexität.
- **XGBoost**: Mächtiger Algorithmus auf Basis von Entscheidungsbäumen mit Boosting.
- **Exponentielle Glättung**: Für glatte Änderungen von Zeitreihendaten.
- **LSTM**: Eine rekurrente neuronale Netzwerkarchitektur, ideal für sequenzbasierte Daten wie Luftqualität.
- **Exponentielle Glättung**: Für glatte Änderungen über die Zeit.

Einige dieser Modelle dienen Lernzwecken.

## Funktionen

- **Kartenvisualisierung**: Airedellín enthält eine interaktive Karte in Echtzeit, die Daten zur Luftqualität von verschiedenen Sensoren in der Stadt anzeigt. Sensoren sind je nach Daten farblich markiert, und beim Klicken auf einen Sensor werden Pop-ups mit detaillierten Informationen angezeigt.
- **Heatmap**: Benutzer können eine Heatmap umschalten, um die Intensität der PM2.5-Konzentrationen in der Stadt zu visualisieren.
- **3D-Relief**: Fügt eine zusätzliche visuelle Ebene hinzu, um die Karte informativer zu gestalten.
- **Prognosen**: Vorhersage von Luftqualitätswerten mit Hilfe von Modellen des maschinellen Lernens.

### Seiten

- `/`: Startseite des Datenvisualisierers.
- `/sensor{sensor_name}`: Zeigt die Sensordaten an und ermöglicht das Filtern nach Datumsbereich.
- `/sensor{sensor_name}/statistics`: Zeigt Sensorstatistiken wie Mittelwert, Varianz und Standardabweichung an.
- `/sensor{sensor_name}/predictions`: Ermöglicht es Benutzern, Prognosealgorithmen auszuwählen und das Ergebnis anzusehen.
- `/add_donation`: Seite für Spenden und Unterstützung von Sensoren.
- `/index`: Indexseite zur Präsentation des Projekts.

## Screenshots 🎑

![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/1.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/3.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/4.png)
![](https://raw.githubusercontent.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin/main/docs/pictures/5.png)

---

## So Wird’s Ausgeführt 🏃‍♀️

Befolgen Sie diese Schritte, um mit Airedellín zu beginnen:

1. **Repository klonen**:

    ```bash
    git clone https://github.com/Semillero-Inteligencia-Artificial-EAFIT/airedellin.git
    cd airedellin
    ```

2. **Umgebung einrichten**:

    - Stellen Sie sicher, dass Sie Python 3.9+ installiert haben.
    - Erstellen Sie eine virtuelle Umgebung:
    
        ```bash
        python -m venv env
        source env/bin/activate  # Auf Windows: `env\Scripts\activate`
        ```

    - Installieren Sie die benötigten Abhängigkeiten:
    
        ```bash
        pip install -r requirements.txt
        ```

    - Fügen Sie Ihr Maptiler-Token in `data/token.txt` ein.

3. **Anwendung starten**:

    ```bash
    uvicorn airedellin:app --host 0.0.0.0 --port 9600
    ```

4. **Weboberfläche aufrufen**:

    Besuchen Sie `http://localhost:9600`, um die Luftqualitätsdaten von Medellín zu erkunden.

---

## Mitwirken 🤝

Airedellín ist ein Open-Source-Projekt, das von der Community getragen wird. Wir schätzen alle Arten von Beiträgen, einschließlich Code, Dokumentation, Design oder Feedback. Zögern Sie nicht, Probleme oder Pull-Requests auf GitHub einzureichen.

---

### Danksagungen 💚

Besonderer Dank an **Hackerspace Unloquer** und **CanAirIO** für die Inspiration und Unterstützung dieses Projekts. Ihr Beitrag zur Verbesserung der Luftqualität in Medellín ist von unschätzbarem Wert!

Schließen Sie sich uns an, um die Luftqualität in Medellín für alle zu verbessern. 🚀🌱

---

### Wichtige Hinweise

- **Daten zu Paris**: Die für Paris verwendeten Daten sind nicht real; sie dienen Testzwecken.

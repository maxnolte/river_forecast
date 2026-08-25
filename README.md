# river_forecast
## Predicting the flow of the river Dranse
[![version](https://img.shields.io/badge/version-0.0.1-yellow.svg)](https://semver.org)

river_forecast is a Python application for predicting the flow of rivers. It is currently limited to the [Dranse](https://en.wikipedia.org/wiki/Dranse_(Haute-Savoie)) in Haute-Savoie, France, a popular river for whitewater kayaking and rafting.

In the standard configuration, river_forecast predicts the flow of the river Dranse in France in real time using gradient boosting (XGBoost). The prediction is currently solely relying on the past water flow. Additional variables such as recent and predicted rainfall will be included in the future. However, even without rain data, the forecast outperforms a naive forecast (see below).

> **Note (2026):** This project is a revival of a model trained six years ago. The XGBoost model was fitted in 2020 on 2016–2020 flow data and is used unchanged — only the data source (now the public Hub'Eau API instead of the discontinued rivermap.ch access) and the deployment (now a self-hosted VPS behind a Cloudflare Tunnel instead of Google Cloud) are new. It still forecasts surprisingly well.

The recent flow data of the Dranse is downloaded from the French Government's public [Hub'Eau hydrometry API](https://hubeau.eaufrance.fr/page/api-hydrometrie) (station V033401001, La Dranse à Reyvroz [Bioge]) — no API key required. Hourly training and test data (01/2016-06/2020) has been provided by the French Government's [Banque Hydro](http://www.hydro.eaufrance.fr/stations/V0334010).

A new prediction is generated every 15 mins. Below is the latest prediction:
![](https://river-forecast.mnolte.fyi/forecast.png)

Link to latest forecast: https://river-forecast.mnolte.fyi/forecast.png


## Model Fitting & Performance

Note that most models were fitted in Google Colab and then transferred and tested in river_forecast. The notebook describing the fitting of the currently best performing model (XGBoost) can be found [here](https://colab.research.google.com/drive/1wB5qWDhYUN9KIuqHfbbg2x_47iNjnetp?usp=sharing).

Below is the full data set. Train and validation sets were used in the Colab for model tuning, test data only for comparison of the final models:

![Train-test split](https://raw.githubusercontent.com/maxnolte/river_forecast/master/images/test-validation-train.png)

The best performing model uses gradient boosting (XGBoost). All models in the error comparison below can currently be used to make predictions:

![Model errors](https://raw.githubusercontent.com/maxnolte/river_forecast/master/images/error-comparison.png)

Below are some example predictions for XGBoost in the test data:

![Prediction examples](https://raw.githubusercontent.com/maxnolte/river_forecast/master/images/dranse-forecast.png)

## Installation

river_forecast has been tested using Anaconda on Ubuntu, Windows, and Windows Subsystem for Linux.

To install the right versions, run the following in a new conda environment:

    conda install --file conda-spec-file.txt

*Warning: currently contains many not required packages.*

## Usage

To generate the current standard prediction, run:

    python bin/create_river_forecast.py

No API key is needed — flow data comes from the public Hub'Eau API.

Other experimental models (SARIMA, LSTM, Naive) can be run by changing the model calls in the script above.

## Deployment

The forecast runs as a one-shot Docker container (`python:3.7-slim`, matching the pickled 2020 model stack) triggered every 15 minutes by a systemd timer. The resulting `forecast.png` is served by nginx on localhost and published through a Cloudflare Tunnel, so the server needs no open inbound ports. See [deploy/DEPLOY.md](deploy/DEPLOY.md) for the full setup.

## Contributing

river_forecast is in its early stages and highly experimental. Please get in touch with me ([maxnolte](https://github.com/maxnolte)) if you have any feedback or suggestions!

## Next Steps

- Include precipitation data.
- Show forecast in dash application.
- Generalize to more rivers from Hub'Eau (Let me know if you have a preference!). 

## License

[MIT](https://choosealicense.com/licenses/mit/)

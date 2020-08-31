# river_forecast
## Predicting the flow of the river Dranse
[![version](https://img.shields.io/badge/version-0.0.1-yellow.svg)](https://semver.org)

river_forecast is a Python application for predicting the flow of rivers. It is currently limited to the [Dranse](https://en.wikipedia.org/wiki/Dranse_(Haute-Savoie)) in Haute-Savoie, France, a popular river for whitewater kayaking and rafting.

In the standard configuration, river_forecast predicts the flow of the river Dranse in France in real time using gradient boosting (XGBoost). The prediction is currently solely relying on the past water flow. Additional variables such as recent and predicted rainfall will be included in the future. However, even without rain data, the forecast outperforms a naive forecast (see below).

The recent flow data of the Dranse is downloaded through the [rivermap.ch](https://www.rivermap.ch/map.html#sprache=en&styled=1&zoom=12&lat=46.33953&lng=6.55717&inf=302) API. Hourly training and test data (01/2016-06/2020) has been provided by the French Government's [Banque Hydro](http://www.hydro.eaufrance.fr/stations/V0334010).

The model is currently deployed on Google Cloud, where a new prediction is generated every 15 mins. Below is the latest prediction:
![](https://storage.googleapis.com/river_forecast/forecast.png)

Link to lastest forecast: https://storage.googleapis.com/river_forecast/forecast.png


## Model Fitting & Performance

Note that most models were fitted in Google Colab and then transferred and tested in river_forecast. Colab notebooks will be made available shortly.

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

*Warning: currently contains many not required packages.

## Usage

To generate the current standard prediction, run:

    python bin/create_river_forecast_local.py

*Warning: a private rivermap.ch API is needed (PRIVATE_RIVERZONE_API_KEY). Get in touch with me ([maxnolte](https://github.com/maxnolte)) if you are interested, and I'll ask the admins of rivermap.ch.
    
Other experimental models (SARIMA, LSTM, Naive) can be run by changing the model calls in the script above.

The model is currently run on a Google Cloud Compute Engine VM using:

    nohup python bin\launch_gcp_compute_engine_schedule.py &

## Contributing

river_forecast is in its early stages and highly experimental. Please get in touch with me ([maxnolte](https://github.com/maxnolte)) if you have any feedback or suggestions!

## License

[MIT](https://choosealicense.com/licenses/mit/)

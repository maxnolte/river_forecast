#!/bin/bash

conda activate river_forecast

cd /home/nolte_max/river_forecast/

python bin/create_river_forecast.py

gsutil cp forecast.png gs://river_forecast/


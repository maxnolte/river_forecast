import schedule
import time
import os


def job():
    os.system("/home/nolte_max/river_forecast/bin/update_forecast_gcp_compute_engine.sh")


schedule.every(15).minutes.do(job)

while 1:
    schedule.run_pending()
    time.sleep(60)

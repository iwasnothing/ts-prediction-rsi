#!/bin/sh

cd /Users/kahingleung/PycharmProjects/timeseries/venv/bin
source activate
cd ../..
python rsi.py
python prophet_rsi.py
tail -1 vgt_rsi_signals.csv >> today_rsi_signals.csv
source source_env.sh
python binance_rsi.py
rm -Rf .darts/checkpoints .darts/untrained_models
chown kahingleung:staff *.sh *.py *.html *.csv *.txt
chown -R kahingleung:staff .git
su - kahingleung -c "/Users/kahingleung/PycharmProjects/timeseries/upload.sh"
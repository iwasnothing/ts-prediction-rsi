#!/bin/sh

cd /Users/kahingleung/PycharmProjects/timeseries/venv/bin
source activate
cd ../..
python rsi.py
rm -Rf .darts/checkpoints .darts/untrained_models
chown kahingleung:staff *.sh *.py *.html *.csv *.txt
chown -R kahingleung:staff .git
su - kahingleung -c "/Users/kahingleung/PycharmProjects/timeseries/upload.sh"
#!/bin/sh

cd /Users/kahingleung/PycharmProjects/timeseries/venv/bin
source activate
cd ../..
python rsi.py
git add *.html *.csv
git commit -am "new signals files"
git push origin main
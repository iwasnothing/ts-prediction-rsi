#!/bin/sh

cd /Users/kahingleung/PycharmProjects/timeseries/venv/bin
source activate
cd ../..
python rsi.py
rm -Rf .darts/checkpoints
su - kahingleung -c "/Users/kahingleung/PycharmProjects/timeseries/upload.sh"
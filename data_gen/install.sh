#!/usr/bin/env bash

cd OpenLabeling
python -mpip install -U pip
sudo python -mpip install -U -r requirements.txt
sudo python -mpip install progressbar
cd ..

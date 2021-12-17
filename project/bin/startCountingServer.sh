#!/bin/bash
export FLASK_APP=project/interface.py
export FLASK_ENV=development
echo "FLASK APP"
echo $FLASK_APP
nohup flask run --host=0.0.0.0 >> log.txt 2>&1 &

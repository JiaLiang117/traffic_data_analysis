#!/bin/bash

echo "Run Train? [Y,n]"
read input
if [[ $input == "Y" || $input == "y" ]]; then
        echo "Running Train";
        python mlp/train.py;
else
        echo "Not Running Train"
fi

echo "Which model to deploy? "
read input

python mlp/deploy.py "models/output.pkl" "https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv" $input
#!/bin/bash

conda activate tfm-mgm

# Array of datasets
datasets=("ili" "weather" "etth1" "etth2" "ettm1" "ettm2")

# Loop through each dataset and execute the Python script
for ds in "${datasets[@]}"; do
    python3 ./experiments/rnn_deep_general_iter.py -ds "$ds"
done

for ds in "${datasets[@]}"; do
    python3 ./experiments/tcn_deep_general_iter.py -ds "$ds"
done
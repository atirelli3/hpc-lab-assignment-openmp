#!/bin/bash

# Set the path to the cholesky.h file
cholesky_file="../cholesky.h"

# Extract dataset names from the file
dataset_names=$(awk '/#.*_DATASET/ && !/#.*!defined/ {print $3}' "$cholesky_file")

# Print the dataset names
echo "Dataset names:"
echo "$dataset_names"

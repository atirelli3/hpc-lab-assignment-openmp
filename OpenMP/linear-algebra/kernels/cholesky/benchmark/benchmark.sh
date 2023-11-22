#!/bin/bash

# Set the path to the cholesky.h file
cholesky_file="cholesky.h"

# Extract dataset names from the file
dataset_names=$(awk '/#.*_DATASET/ && !/#.*!defined/ {print $3}' "$cholesky_file")

# Iterate over dataset names and execute the command
for dataset_name in $dataset_names; do
    echo "Executing for dataset: $dataset_name"
    make EXT_CFLAGS="-DPOLYBENCH_TIME -D${dataset_name} -DPARALLEL_OPT" clean all
done

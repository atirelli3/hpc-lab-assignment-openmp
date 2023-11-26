#!/bin/bash

cd ..

# Set the path to the cholesky.h file
cholesky_file="cholesky.h"

# Extract dataset names from the file
dataset_names=$(awk '/#.*_DATASET/ && !/#.*!defined/ {print $3}' "$cholesky_file")

# Function to run with perf stat and suppress other outputs
run_with_perf() {
    # make EXT_CFLAGS="$1 -DPOLYBENCH_DUMP_ARRAYS" clean all > /dev/null 2>&1
    make EXT_CFLAGS="$1" clean all
    perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations ./cholesky_acc
}

# Iterate over dataset names and execute the command
for dataset_name in $dataset_names; do
    echo "### Executing for dataset: $dataset_name"

    # SEQUENTIAL
    echo "=> $dataset_name SEQUENTIAL"
    run_with_perf "-D${dataset_name}"
    # ./cholesky_acc > out_seq.txt 2>&1
    ./cholesky_acc

    # PARALLEL_OPT
    echo "=> $dataset_name PARALLEL_OPT"
    run_with_perf "-D${dataset_name} -DPARALLEL_OPT"
    # ./cholesky_acc > out_opt.txt 2>&1
    ./cholesky_acc

    # Compare the output files
    # echo "### Diff between out_seq.txt and out_opt.txt:"
    # diff -u -q out_seq.txt out_opt.txt
done

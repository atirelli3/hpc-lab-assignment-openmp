#!/bin/bash

cd ..

# Set the path to the cholesky.h file
cholesky_file="cholesky.h"

# Extract dataset names from the file
dataset_names=$(awk '/#.*_DATASET/ && !/#.*!defined/ {print $3}' "$cholesky_file")

compile_with_dataset() {
    make EXT_CFLAGS="$1" clean all > /dev/null 2>&1
}

compile_with_arrays() {
    make EXT_CFLAGS="$1 -DPOLYBENCH_DUMP_ARRAYS" clean all > /dev/null 2>&1
}

# Function to run with perf stat and suppress other outputs
run_with_perf() {
    perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations ./cholesky_acc
}

# Iterate over dataset names and execute the command
for dataset_name in $dataset_names; do
    echo "####### Executing for dataset: $dataset_name #######"

    # SEQUENTIAL
    echo "@@@@@ => $dataset_name SEQUENTIAL"
    # compile_with_dataset "-D${dataset_name}"
    compile_with_arrays "-D${dataset_name}"
    # run_with_perf
    ./cholesky_acc > out_seq.txt 2>&1

    # MEM_OPT
    # echo "@@@@@ => $dataset_name MEM_OPT"
    # compile_with_dataset "-D${dataset_name} -DMEM_OPT"
    compile_with_arrays "-D${dataset_name} -DMEM_OPT"
    # run_with_perf
    ./cholesky_acc > out_mem.txt 2>&1

    # PARALLEL_OPT
    # echo "@@@@@ => $dataset_name PARALLEL_OPT"
    # complie_with_dataset "-D${dataset_name} -DPARALLEL_OPT"
    # compile_with_arrays "-D${dataset_name} -DPARALLEL_OPT"
    # run_with_perf
    # ./cholesky_acc > out_opt.txt 2>&1

    # Compare the output files
    # echo "### Diff between out_seq.txt and out_opt.txt:"
    # echo "### Diff between out_seq.txt and out_mem.txt:"
    # diff -q out_seq.txt out_mem.txt
    echo "###################################################"
done


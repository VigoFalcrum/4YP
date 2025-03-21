#!/bin/bash
# run_models.sh
# This script iterates over different RF model configurations and runs them for multiple core counts. 
# For each configuration and each core count (1, 2, 4, 8, 16), it runs RF_train.py exactly once. 
# RF_train.py must accept two command-line arguments: n_estimators, max_depth and min_samples_split, 
# and it must print a single number (the p50 online latency).
# The overall summary is saved in overall_results.txt.

# Define the model configurations (60 configurations total).
n_estimators=(2 2 2 2 2 2 2 2 2 2 5 5 5 5 5 5 5 5 5 5 15 15 15 15 15 15 15 15 15 15 70 70 70 70 70 70 70 70 70 70 \
200 200 200 200 200 200 200 200 200 200 500 500 500 500 500 500 500 500 500 500)
max_depths=(7 7 11 11 15 15 20 20 30 30 7 7 11 11 15 15 20 20 30 30 7 7 11 11 15 15 20 20 30 30 7 7 11 11 15 15 20 20 30 30 \
7 7 11 11 15 15 20 20 30 30)
min_samples_split=(1000 250 250 100 100 60 5 2 5 2 500 150 100 70 30 20 5 2 5 2 250 100 70 30 15 10 5 2 5 2 100 30 30 10 5 2 5 2 5 2 \
30 10 15 7 5 2 5 2 5 2 5 2 5 2 5 2 5 2 5 2)

# Define the core counts to test.
core_counts=(1 2 4 8 16)

# Overall results file for summary.
overall_results="overall_testing_results.txt"
> "$overall_results"
echo "n_estimators	max_depths	min_samples_split	cores" >> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
	n_estimators=${n_estimators[$i]}
        max_depth=${max_depths[$i]}
        min_samples_split=${min_samples_split[$i]}
        echo "Running model with max_depth=${max_depth}, min_samples_split=${min_samples_split}"
        
        # Run RF_train.py once with the specified core count.
        latency=$(OMP_NUM_THREADS=$core python3 RF_train.py "$n_estimators" "$max_depth" "$min_samples_split")
        
        # Append the result to overall_results.txt.
        echo "${latency} ${n_estimators} ${max_depth} ${min_samples_split} ${core}" >> $overall_results
        
        echo "Completed model with n_estimators=${n_estimators}, max_depth=${max_depth}, min_samples_split=${min_samples_split} for ${core} core(s)"
        echo "-------------------------------------------------------"
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."
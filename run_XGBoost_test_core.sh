#!/bin/bash
# run_models.sh
# This script iterates over different RF model configurations and runs them for multiple core counts. 
# For each configuration and each core count (1, 2, 4, 8, 16), it runs RF_train.py exactly once. 
# RF_train.py must accept two command-line arguments: n_estimators, max_depth and min_samples_split, 
# and it must print a single number (the p50 online latency).
# The overall summary is saved in overall_results.txt.

# Define the model configurations.
n_estimators=(10 10 10 50 50 50 100 100 100 150 150 150 250 250 250 600 600 600 1000 1000 1000)
max_depths=(2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9)

# Define the core counts to test.
core_counts=(1 2 4 8 16 32)

# Overall results file for summary.
overall_results="XGBoost_testing_results.txt"
> "$overall_results"
echo "latency	n_estimators	max_depths	cores" >> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
	n_estimators_v=${n_estimators[$i]}
        max_depth_v=${max_depths[$i]}
        echo "Running model with n_estimators=${n_estimators_v}, max_depth=${max_depth_v}"
        
        # Run RF_train.py once with the specified core count.
        latency=$(OMP_NUM_THREADS=1 python3 XGBoost_test.py "$n_estimators_v" "$max_depth_v" "$core")
        
        # Append the result to overall_results.txt.
        echo "${latency} ${n_estimators_v} ${max_depth_v} ${core}" >> $overall_results
        
        echo "Completed model with n_estimators=${n_estimators_v}, max_depth=${max_depth_v} for ${core} core(s) in ${latency} seconds"
        echo "-------------------------------------------------------"
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."
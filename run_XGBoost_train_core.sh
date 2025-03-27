#!/bin/bash
# run_models.sh
# This script iterates over different XGBoost model configurations and runs them for multiple core counts. 
# For each configuration and each core count (1, 2, 4, 8, 16), it runs RF_train.py exactly once. 
# XGBoost_train.py must accept two command-line arguments: n_estimators, max_depth and learning_rate, 
# and it must print a single number (the p50 online latency).
# The overall summary is saved in overall_results.txt.

# Define the model configurations.
n_estimators=(10 10 10 50 50 50 100 100 100 150 150 150 250 250 250 600 600 600 1000 1000 1000)
max_depths=(2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9)
learning_rate=(0.3 0.16 0.04 0.2 0.12 0.03 0.15 0.08 0.02 0.1 0.05 0.016 0.04 0.03 0.012 0.02 0.01 0.006 0.01 0.008 0.002)

# Define the core counts to test.
core_counts=(1 2 4 8 16 32)

# Overall results file for summary.
overall_results="overall_testing_results.txt"
> "$overall_results"
echo "latency n_estimators max_depth learning_rate cores" >> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
        # Hardcode the parameters for i=0 due to an unexplained bug
        if [ "$i" -eq 0 ]; then
            n_estimators=10
            max_depth=2
            learning_rate=0.3
        else
	    n_estimators=${n_estimators[$i]}
            max_depth=${max_depths[$i]}
            learning_rate=${learning_rate[$i]}
        fi
        echo "Running model with n_estimators=${n_estimators}, max_depth=${max_depth}, learning_rate=${learning_rate}"
        
        # Run XGBoost_train.py once with the specified core count.
        latency=$(OMP_NUM_THREADS=$core python3 XGBoost_train.py "$n_estimators" "$max_depth" "$learning_rate" "$core")
        
        # Append the result to overall_results.txt.
        echo "${latency} ${n_estimators} ${max_depth} ${learning_rate} ${core}" >> $overall_results
        
        echo "Completed model with n_estimators=${n_estimators}, max_depth=${max_depth}, learning_rate=${learning_rate} for ${core} core(s) in ${latency} seconds"
        echo "-------------------------------------------------------"
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."
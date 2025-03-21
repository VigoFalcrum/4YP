#!/bin/bash
# run_models.sh
# This script iterates over different Decision Tree model configurations and runs them for multiple core counts.
# For each configuration and each core count (1, 2, 4, 8, 16), it runs DT_test.py exactly once.
# DT_test.py must accept two command-line arguments: max_depth and min_samples_split,
# and it must print a single number (the p50 online latency).
# The overall summary is saved in overall_results.txt.

# Define the model configurations (13 configurations total).
max_depths=(3 7 7 10 10 20 20 25 25 30 30 None None)
min_samples=(250 1000 250 70 15 70 15 5 2 5 2 5 2)

# Define the core counts to test.
core_counts=(1 2 4 8 16)

# Overall results file for summary.
overall_results="overall_testing_results.txt"
> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do
    echo "===== Running tests with ${core} core(s) =====" | tee -a "$overall_results"
    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
        max_depth=${max_depths[$i]}
        min_samples_split=${min_samples[$i]}
        echo "Running model with max_depth=${max_depth}, min_samples_split=${min_samples_split}" | tee -a "$overall_results"
        
        # Run DT_test.py once with the specified core count.
        latency=$(OMP_NUM_THREADS=$core python3 DT_test.py "$max_depth" "$min_samples_split")
        
        # Append the result to overall_results.txt.
        {
          echo "Configuration: max_depth=${max_depth}, min_samples_split=${min_samples_split}, Cores=${core}"
          echo "Latency (p50 online): ${latency}"
          echo "-------------------------------------------------------"
        } >> "$overall_results"
        
        echo "Completed model with max_depth=${max_depth}, min_samples_split=${min_samples_split} for ${core} core(s)"
        echo "-------------------------------------------------------"
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."

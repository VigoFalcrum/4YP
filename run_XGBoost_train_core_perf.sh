#!/bin/bash
# run_models.sh
# This script iterates over different RF model configurations and runs them for multiple core counts.
# For each configuration and each core count (1,2,4,8,16,32), it runs RF_train_perf.py with perf stat,
# extracts the training latency, cache misses, the percentage of cache references missed, CPUs utilized, and instructions per cycle (IPC),
# and then appends a summary line to overall_testing_results.txt.

# Define the model configurations.
n_estimators=(10 10 10 50 50 50 100 100 100 150 150 150 250 250 250 600 600 600 1000 1000 1000 2000 2000 2000)
max_depths=(2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9 2 5 9)
learning_rate=(0.3 0.16 0.04 0.2 0.12 0.03 0.15 0.08 0.02 0.1 0.05 0.016 0.04 0.03 0.012 0.02 0.01 0.006 0.01 0.008 0.002 0.008 0.005 0.001)

# Define the core counts to test.
core_counts=(1 2 4 8 16 32)

# Overall results file for summary.
overall_results="XGBoost_training_perf_results.txt"
> "$overall_results"
# Update header to include new metric: cache_miss_percent.
echo "n_estimators,max_depth,learning_rate,cores,training_time,cache_miss_percent,cpus_utilized,ipc" >> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
        n_estimators_val=${n_estimators[$i]}
        max_depth_val=${max_depths[$i]}
        learning_rate_val=${learning_rate[$i]}
        echo "Running model with n_estimators=${n_estimators_val}, max_depth=${max_depth_val}, learning_rate=${learning_rate_val}, cores=${core}" >&2
        
        # Run XGBoost_train_perf.py once with the specified core count.
        # We run perf stat to capture cache-misses, task-clock, instructions, and cycles.
        # We redirect stderr to stdout (2>&1) so that all output is captured.
        perf_output=$(OMP_NUM_THREADS=$core sudo -E perf stat -e cache-misses,cache-references,task-clock,instructions,cycles \
            python3 XGBoost_train_perf.py --train "$n_estimators_val" "$max_depth_val" "$learning_rate_val" "$core" 2>&1)
        
        # Extract training time.
        # Assumes XGBoost_train_perf.py prints "Training time: <number> seconds"
        training_time=$(echo "$perf_output" | grep -i "Training time:" | awk '{print $3}')
                
        # Parse the percentage of cache references missed.
        # This assumes that the cache-misses line has a fragment like "#   46.160 % of all cache refs"
        cache_percent=$(echo "$perf_output" | grep "cache-misses" | awk -F'#' '{print $2}' | awk '{print $1}' | tr -d '%')
        
        # Extract CPUs utilized from the task-clock line.
        # Looks for a comment field such as "#   29.085 CPUs utilized"
        cpus_utilized=$(echo "$perf_output" | grep "task-clock" | awk -F'#' '{print $2}' | awk '{print $1}')
        
        # Extract instructions and cycles (removing commas)
        instructions=$(echo "$perf_output" | grep "instructions" | head -n1 | awk '{print $1}' | tr -d ',')
        cycles=$(echo "$perf_output" | grep "cycles" | head -n1 | awk '{print $1}' | tr -d ',')
        
        # Calculate IPC = instructions / cycles.
        if [ -n "$cycles" ] && [ "$cycles" -gt 0 ]; then
          ipc=$(awk "BEGIN {printf \"%.2f\", $instructions/$cycles}")
        else
          ipc="N/A"
        fi
        
        # Append the result to the CSV file.
        echo "${n_estimators_val},${max_depth_val},${learning_rate_val},${core},${training_time},${cache_percent},${cpus_utilized},${ipc}" >> "$overall_results"
        
        echo "Completed model: n_estimators=${n_estimators_val}, max_depth=${max_depth_val}, learning_rate=${learning_rate_val}, cores=${core} in ${training_time} seconds" >&2
        echo "-------------------------------------------------------" >&2
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."

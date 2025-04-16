#!/bin/bash
# run_models.sh
# This script iterates over different RF model configurations and runs them for multiple core counts.
# For each configuration and each core count (1,2,4,8,16,32), it runs RF_train_perf.py with perf stat,
# extracts the training latency, cache misses, the percentage of cache references missed, CPUs utilized, and instructions per cycle (IPC),
# and then appends a summary line to overall_testing_results.txt.

# Define the model configurations.
n_estimators=(5 5 5 15 15 15 45 45 45 90 90 90 140 140 140 200 200 200 270 270 270)
max_depths=(7 15 25 7 15 25 7 15 25 7 15 25 7 15 25 7 15 25 7 15 25)
min_samples_split=(150 20 2 100 10 2 50 5 2 25 2 2 15 2 2 10 2 2 6 2 2)

# Define the core counts to test.
core_counts=(1 2 4 8 16 32)

# Overall results file for summary.
overall_results="RF_testing_perf_results.txt"
> "$overall_results"
# Update header to include new metric: cache_miss_percent.
echo "n_estimators,max_depth,min_samples_split,cores,training_time,cache_miss_percent,cpus_utilized,ipc" >> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
        n_estimators_val=${n_estimators[$i]}
        max_depth_val=${max_depths[$i]}
        min_samples_split_val=${min_samples_split[$i]}
        echo "Running model with n_estimators=${n_estimators_val}, max_depth=${max_depth_val}, min_samples_split=${min_samples_split_val}, cores=${core}" >&2
        
        # Run RF_train_perf.py once with the specified core count.
        # We run perf stat to capture cache-misses, task-clock, instructions, and cycles.
        # We redirect stderr to stdout (2>&1) so that all output is captured.
        perf_output=$(OMP_NUM_THREADS=$core sudo -E perf stat -e cache-misses,cache-references,task-clock,instructions,cycles \
            python3 RF_test_perf.py --test "$n_estimators_val" "$max_depth_val" "$min_samples_split_val" "$core" 2>&1)
        
        # Extract testing time.
        # Assumes RF_train_perf.py prints "Training time: <number> seconds"
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
        echo "${n_estimators_val},${max_depth_val},${min_samples_split_val},${core},${training_time},${cache_percent},${cpus_utilized},${ipc}" >> "$overall_results"
        
        echo "Completed model: n_estimators=${n_estimators_val}, max_depth=${max_depth_val}, min_samples_split=${min_samples_split_val}, cores=${core} in ${training_time} seconds" >&2
        echo "-------------------------------------------------------" >&2
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."

#!/bin/bash
# run_models.sh
# This script iterates over different RF model configurations and runs them for multiple core counts.
# For each configuration and each core count (1,2,4,8,16,32), it runs RF_train_perf.py with perf stat,
# extracts the training latency, cache misses, the percentage of cache references missed, CPUs utilized, and instructions per cycle (IPC),
# and then appends a summary line to overall_testing_results.txt.

# Define the model configurations.
# depths=(2 3 5 7 9 2 3 5 7 9 2 3 5 7 9 2 3 5 7 9 2 3 5 7 9 2 3 5 7 9 2 3 5 7 9 2 3 5 7 9)
# hidden_sizes=(4 4 4 4 4 8 8 8 8 8 16 16 16 16 16 32 32 32 32 32 64 64 64 64 64 128 128 128 128 128 256 256 256 256 256 512 512 512 512 512)
depths=(2 3 5 7 9 2 3 5 7 9)
hidden_sizes=(1024 1024 1024 1024 1024 2048 2048 2048 2048 2048)

# Define the core counts to test.
core_counts=(32)

# Overall results file for summary.
overall_results="NN_training_perf_results_high.txt"
> "$overall_results"
# Update header to include new metric: cache_miss_percent.
echo "max_depth,hidden_size,cores,training_time,cache_miss_percent,cpus_utilized,ipc" >> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do    
    num_configs=${#depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
        hidden_size_val=${hidden_sizes[$i]}
        depth_val=${depths[$i]}
 
        echo "Running model with depth=${depth_val}, hidden_size=${hidden_size_val}, cores=${core}" >&2
        
        # Run RF_train_perf.py once with the specified core count.
        # We run perf stat to capture cache-misses, task-clock, instructions, and cycles.
        # We redirect stderr to stdout (2>&1) so that all output is captured.
        perf_output=$(OMP_NUM_THREADS=$core sudo -E perf stat -e cache-misses,cache-references,task-clock,instructions,cycles \
            python3 NN_train_perf.py --train "$depth_val" "$hidden_size_val" "$core" 2>&1)
        
        # Extract training time.
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
        echo "${depth_val},${hidden_size_val},${core},${training_time},${cache_percent},${cpus_utilized},${ipc}" >> "$overall_results"
        
        echo "Completed model: depth=${depth_val}, hidden_size=${hidden_size_val}, cores=${core} in ${training_time} seconds" >&2
        echo "-------------------------------------------------------" >&2
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."

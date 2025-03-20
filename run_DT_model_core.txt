#!/bin/bash
# run_models.sh
# This script iterates over different Decision Tree model configurations and runs them for multiple core counts.
# For each configuration and each core count (1, 2, 4, 8, 16), it repeatedly runs DT_model.py and collects latency measurements.
# It computes the mean, standard deviation, and 95% confidence interval margin of error (MoE) (relative to the mean).
# Once the MoE for latency falls below 3%, it logs the aggregated results for that configuration and core count.
#
# Requirements:
# - DT_model.py must accept two command-line arguments: max_depth and min_samples_split,
#   and it must print a single number (the latency, as measured by time.perf_counter()).
# - It is assumed that setting OMP_NUM_THREADS will control the number of cores used by DT_model.py.
# - This script uses awk (with sqrt) and bc, so it assumes you have GNU awk and bc installed.
#
# Intermediate results are not stored permanently; only the overall summary is saved in overall_results.txt.

# Define the model configurations (13 configurations total).
max_depths=(3 7 7 10 10 20 20 25 25 30 30 None None)
min_samples=(250 1000 250 70 15 70 15 5 2 5 2 5 2)

# Define the core counts to test.
core_counts=(1 2 4 8 16)

# Overall results file for summary.
overall_results="overall_results.txt"
> "$overall_results"

# Outer loop: iterate over the specified core counts.
for core in "${core_counts[@]}"; do
    echo "===== Running tests with ${core} core(s) =====" >> "$overall_results"
    echo "===== Running tests with ${core} core(s) ====="
    
    num_configs=${#max_depths[@]}
    # Iterate over each model configuration.
    for (( i=0; i<num_configs; i++ )); do
        max_depth=${max_depths[$i]}
        min_samples_split=${min_samples[$i]}
        echo "Running model with max_depth=${max_depth}, min_samples_split=${min_samples_split}" >> "$overall_results"
        echo "Running model with max_depth=${max_depth}, min_samples_split=${min_samples_split}"
        
        # Create a temporary file to store latency measurements.
        temp_file=$(mktemp)
        
        iteration=0
        min_iterations=10  # Minimum iterations before checking MoE.
        moe=100  # Initialize MoE.
        
        # Loop until the latency MoE falls below 3%.
        while true; do
            iteration=$(( iteration + 1 ))
            
            # Run DT_model.py with the specified core count.
            # OMP_NUM_THREADS is used to control the number of cores.
            latency=$(OMP_NUM_THREADS=$core python3 DT_model.py "$max_depth" "$min_samples_split")
            
            # Append the latency value to the temporary file.
            echo "$latency" >> "$temp_file"
            
            # Once we have enough iterations, calculate the MoE.
            if (( iteration >= min_iterations )); then
                moe=$(awk '{
                    sum += $1;
                    sumsq += ($1)^2;
                    count++;
                }
                END {
                    if(count > 0) {
                        mean = sum / count;
                        std = sqrt(sumsq/count - mean^2);
                        moe = 1.96 * std / sqrt(count) / mean * 100;
                        printf "%.6f", moe;
                    }
                }' "$temp_file")
                
                echo "Iteration: $iteration, Current Latency MoE: ${moe}%"
                
                # Stop if MoE falls below 3%.
                if [ $(echo "$moe < 3" | bc -l) -eq 1 ]; then
                    break
                fi
            fi
        done
        
        # Compute final aggregated statistics for latency.
        stats=$(awk '{
            sum += $1;
            sumsq += ($1)^2;
            count++;
        }
        END {
            if(count > 0) {
                mean = sum / count;
                std = sqrt(sumsq/count - mean^2);
                moe = 1.96 * std / sqrt(count) / mean * 100;
                printf "Iterations: %d\nLatency: Mean=%.6f, Std=%.6f\nLatency MoE: %.6f%%\n", count, mean, std, moe;
            }
        }' "$temp_file")
        
        # Append the summary for this configuration and core count to overall_results.txt.
        {
          echo "Configuration: max_depth=${max_depth}, min_samples_split=${min_samples_split}, Cores=${core}"
          echo "$stats"
          echo "-------------------------------------------------------"
        } >> "$overall_results"
        
        # Remove the temporary file.
        rm "$temp_file"
        
        echo "Completed model with max_depth=${max_depth}, min_samples_split=${min_samples_split} for ${core} core(s)"
        echo "-------------------------------------------------------"
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."

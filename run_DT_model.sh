#!/bin/bash
# run_models.sh
# This script iterates over different Decision Tree model configurations.
# For each configuration, it repeatedly runs train_script.py and collects latency measurements.
# It computes the mean, standard deviation, and 95% confidence interval margin of error (MoE)
# (relative to the mean). Once the MoE falls below 3%, it moves to the next configuration.
#
# Requirements:
# - train_script.py must accept two command-line arguments: max_depth and min_samples_split,
#   and it must print a single number (the latency, as measured by time.perf_counter()).
# - This script uses awk (with sqrt) and bc, so it assumes you have GNU awk and bc installed.

# Define the model configurations in two arrays (13 configurations total).
max_depths=(3 7 7 10 10 20 20 25 25 30 30 None None)
min_samples=(250 1000 250 70 15 70 15 5 2 5 2 5 2)

# Overall results file for summary of each configuration.
overall_results="overall_results.txt"
> "$overall_results"

# Loop over each configuration.
num_configs=${#max_depths[@]}
for (( i=0; i<num_configs; i++ )); do
    max_depth=${max_depths[$i]}
    min_samples_split=${min_samples[$i]}
    echo "Running model with max_depth=${max_depth}, min_samples_split=${min_samples_split}"
    
    # Create (or clear) a results file for this configuration.
    result_file="results_maxdepth_${max_depth}_minsplit_${min_samples_split}.txt"
    > "$result_file"
    
    iteration=0
    moe=100   # Start with a high MoE value.
    min_iterations=10  # Wait until at least 10 measurements before checking MoE.
    
    # Loop until the MoE (relative to mean latency) falls below 3%.
    while true; do
        iteration=$(( iteration + 1 ))
        
        # Run the python training script.
        # It must output a single numeric latency value.
        latency=$(python3 DT_model.py "$max_depth" "$min_samples_split")
        
        # Append the latency value to the results file.
        echo "$latency" >> "$result_file"
        
        # If we have enough samples, compute statistics.
        if (( iteration >= min_iterations )); then
            # Using awk to compute: count, mean, std, and relative MoE (in %).
            # MoE is computed as: 1.96 * std / sqrt(n) / mean * 100
            stats=$(awk '
            {
                sum += $1;
                sumsq += ($1)^2;
                count++
            }
            END {
                if(count > 0) {
                    mean = sum / count;
                    std = sqrt(sumsq/count - mean^2);
                    moe = 1.96 * std / sqrt(count) / mean * 100;
                    printf "%d %.6f %.6f %.6f", count, mean, std, moe;
                }
            }' "$result_file")
            
            # Parse the statistics.
            count=$(echo $stats | awk '{print $1}')
            mean=$(echo $stats | awk '{print $2}')
            std=$(echo $stats | awk '{print $3}')
            moe=$(echo $stats | awk '{print $4}')
            
            echo "Iteration: $iteration, Count: $count, Mean: $mean, Std: $std, MoE: ${moe}%"
            
            # Check if the relative MoE is below 3%.
            if [ $(echo "$moe < 3" | bc -l) -eq 1 ]; then
                echo "Achieved MoE < 3% for model max_depth=${max_depth}, min_samples_split=${min_samples_split}"
                echo "Configuration: max_depth=${max_depth}, min_samples_split=${min_samples_split}, iterations=$count, mean=$mean, std=$std, moe=${moe}%" >> "$overall_results"
                break
            fi
        fi
    done
    echo "Results for model (max_depth=${max_depth}, min_samples_split=${min_samples_split}) stored in ${result_file}"
    echo "-------------------------------------------------------"
done

echo "All configurations completed. Summary stored in ${overall_results}."

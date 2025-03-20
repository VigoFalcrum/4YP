#!/bin/bash
# run_models.sh
# This script iterates over different Decision Tree model configurations.
# For each configuration, it repeatedly runs the DT_model.py script via perf stat
# to collect latency and performance counter metrics. It aggregates the results
# (mean and standard deviation for each metric) and logs them in overall_results.txt.
#
# The DT_model.py script must accept two command-line arguments: max_depth and min_samples_split,
# and it must print a single numeric latency value.
#
# Note: We do not save individual run results permanently; a temporary file is used per configuration.
#
# Define the model configurations.
max_depths=(3 7 7 10 10 20 20 25 25 30 30 None None)
min_samples=(250 1000 250 70 15 70 15 5 2 5 2 5 2)

# Overall results file for summary.
overall_results="overall_results.txt"
> "$overall_results"

num_configs=${#max_depths[@]}
for (( i=0; i<num_configs; i++ )); do
    max_depth=${max_depths[$i]}
    min_samples_split=${min_samples[$i]}
    echo "Running model with max_depth=${max_depth}, min_samples_split=${min_samples_split}"
    
    # Create a temporary file to hold iteration results.
    temp_file=$(mktemp)
    
    iteration=0
    min_iterations=10  # Minimum iterations before checking MoE.
    moe=100  # Initial (dummy) MoE value.
    
    while true; do
        iteration=$(( iteration + 1 ))
        
        # Run the perf stat command and capture output.
        # We redirect stderr (where perf writes its counters) along with stdout.
        perf_output=$(sudo perf stat -e l2_pf_miss_l2_l3,l2_pf_miss_l2_hit_l3,stalled-cycles-backend,instructions,cycles python3 DT_model.py "$max_depth" "$min_samples_split" 2>&1)
        
        # Extract the latency.
        # Assume DT_model.py prints a latency value as the first non-empty line.
        latency=$(echo "$perf_output" | grep -E '^[0-9]' | head -n1 | tr -d ',')
        
        # Extract the perf counter values by grepping for their event names.
        l2_pf_miss_l2_l3=$(echo "$perf_output" | grep "l2_pf_miss_l2_l3" | awk '{print $1}' | tr -d ',')
        l2_pf_miss_l2_hit_l3=$(echo "$perf_output" | grep "l2_pf_miss_l2_hit_l3" | awk '{print $1}' | tr -d ',')
        stalled_cycles_backend=$(echo "$perf_output" | grep "stalled-cycles-backend" | awk '{print $1}' | tr -d ',')
        instructions=$(echo "$perf_output" | grep "instructions" | awk '{print $1}' | tr -d ',')
        cycles=$(echo "$perf_output" | grep "cycles" | awk '{print $1}' | tr -d ',')
        
        # Ensure all values were captured.
        if [ -z "$latency" ] || [ -z "$l2_pf_miss_l2_l3" ] || [ -z "$l2_pf_miss_l2_hit_l3" ] || [ -z "$stalled_cycles_backend" ] || [ -z "$instructions" ] || [ -z "$cycles" ]; then
            echo "Error parsing perf output; skipping iteration $iteration" >&2
            continue
        fi
        
        # Append one line (six space-separated numbers) to the temporary file.
        # Columns: latency, l2_pf_miss_l2_l3, l2_pf_miss_l2_hit_l3, stalled-cycles-backend, instructions, cycles.
        echo "$latency $l2_pf_miss_l2_l3 $l2_pf_miss_l2_hit_l3 $stalled_cycles_backend $instructions $cycles" >> "$temp_file"
        
        # Once we have enough iterations, compute MoE for latency (column 1).
        if (( iteration >= min_iterations )); then
            moe=$(awk '{
                sum += $1;
                sumsq += ($1)^2;
                count++;
            }
            END {
                if(count > 0){
                    mean = sum / count;
                    std = sqrt(sumsq/count - mean^2);
                    moe = 1.96 * std / sqrt(count) / mean * 100;
                    printf "%.6f", moe;
                }
            }' "$temp_file")
            
            echo "Iteration: $iteration, Current Latency MoE: ${moe}%"
            
            # If the MoE for latency is below 3%, break out of the loop.
            if [ $(echo "$moe < 3" | bc -l) -eq 1 ]; then
                break
            fi
        fi
    done
    
    # Compute aggregated statistics (mean and std) for all metrics from the temporary file.
    stats=$(awk '{
        for(i=1; i<=6; i++){
            sum[i] += $i;
            sumsq[i] += ($i)^2;
        }
        count++;
    }
    END {
        if(count>0){
            mean_latency = sum[1]/count; std_latency = sqrt(sumsq[1]/count - mean_latency^2);
            mean_l2_pf_miss_l2_l3 = sum[2]/count; std_l2_pf_miss_l2_l3 = sqrt(sumsq[2]/count - mean_l2_pf_miss_l2_l3^2);
            mean_l2_pf_miss_l2_hit_l3 = sum[3]/count; std_l2_pf_miss_l2_hit_l3 = sqrt(sumsq[3]/count - mean_l2_pf_miss_l2_hit_l3^2);
            mean_stalled = sum[4]/count; std_stalled = sqrt(sumsq[4]/count - mean_stalled^2);
            mean_instructions = sum[5]/count; std_instructions = sqrt(sumsq[5]/count - mean_instructions^2);
            mean_cycles = sum[6]/count; std_cycles = sqrt(sumsq[6]/count - mean_cycles^2);
            moe_latency = 1.96 * std_latency / sqrt(count) / mean_latency * 100;
            printf "Iterations: %d\nLatency: Mean=%.6f, Std=%.6f\nl2_pf_miss_l2_l3: Mean=%.6f, Std=%.6f\nl2_pf_miss_l2_hit_l3: Mean=%.6f, Std=%.6f\nstalled-cycles-backend: Mean=%.6f, Std=%.6f\ninstructions: Mean=%.6f, Std=%.6f\ncycles: Mean=%.6f, Std=%.6f\nLatency MoE: %.6f%%\n", count, mean_latency, std_latency, mean_l2_pf_miss_l2_l3, std_l2_pf_miss_l2_l3, mean_l2_pf_miss_l2_hit_l3, std_l2_pf_miss_l2_hit_l3, mean_stalled, std_stalled, mean_instructions, std_instructions, mean_cycles, std_cycles, moe_latency;
        }
    }' "$temp_file")
    
    {
      echo "Configuration: max_depth=${max_depth}, min_samples_split=${min_samples_split}"
      echo "$stats"
      echo "-------------------------------------------------------"
    } >> "$overall_results"
    
    # Remove the temporary file.
    rm "$temp_file"
    
    echo "Completed model with max_depth=${max_depth}, min_samples_split=${min_samples_split}"
done

echo "All configurations completed. Summary stored in ${overall_results}."

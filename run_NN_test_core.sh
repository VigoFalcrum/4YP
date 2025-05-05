#!/bin/bash
# run_nn.sh
# This script runs NN_train.py for each combination of parameters:
#   - depth: (1, 3, 5, 7)
#   - hidden layer size: (32, 128, 512, 1024, 2048)
#   - core count: (1, 2, 4, 8, 16, 32)
#
# NN_train.py should accept three command-line arguments:
#   depth, hidden_layer_size, core_count
# and print out a single number representing training latency.
#
# The output is logged in nn_results.txt in a tab-separated format.

# Define parameter arrays.
depths=(2 5 7 9)
hidden_sizes=(4 8 16 32 64 128 256 512 1024 1024)
cores=(32)

# Define output file and write header.
results_file="NN_online_test_results.txt"
> "$results_file"
echo -e "depth\thidden_size\tcore_count\tlatency" >> "$results_file"

# Iterate over each parameter combination.
for depth in "${depths[@]}"; do
    for hidden in "${hidden_sizes[@]}"; do
        for core in "${cores[@]}"; do
            echo "Running NN with depth=${depth}, hidden_size=${hidden}, core_count=${core}"
            
            # Run the training script and capture the output.
            latency=$(python3 NN_online_test.py --test "$hidden" "$depth" "$core")
            
            # Append the parameters and resulting latency to the results file.
            echo -e "${depth}\t${hidden}\t${core}\t${latency}" >> "$results_file"
            
            echo "Completed NN with depth=${depth}, hidden_size=${hidden}, core_count=${core}: latency=${latency} seconds"
            echo "-------------------------------------------------------"
        done
    done
done

echo "All configurations completed. Results stored in ${results_file}."

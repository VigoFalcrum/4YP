#!/bin/bash
# run_models.sh
# This script iterates over different dataset fractions and runs SVM.py for each fraction.
# SVM.py must accept one command-line argument: the fraction (a float)
# and it must print three lines:
#   1. training latency
#   2. testing latency
#   3. accuracy
# The overall summary is saved in SVM_testing_results.txt.

# Define the dataset fractions.
frac=(0.001 0.002 0.004)

# Overall results file for summary.
overall_results="SVM_testing_results.txt"
> "$overall_results"
echo -e "frac\ttraining_time\ttesting_time\taccuracy" >> "$overall_results"

# Iterate over each fraction.
num_configs=${#frac[@]}
for (( i=0; i<num_configs; i++ )); do
    frac_v=${frac[$i]}
    echo "Running model with frac=${frac_v}"
    
    # Run SVM.py with the fraction as an argument and capture its output.
    output=$(python3 SVM.py "$frac_v")
    
    # Read the three lines of output into variables.
    IFS=$'\n' read -r training_time testing_time accuracy <<< "$output"
    
    # Append a single line to the overall results file.
    echo -e "${frac_v}\t${training_time}\t${testing_time}\t${accuracy}" >> "$overall_results"
    
    echo "Completed model with frac=${frac_v}: training_time=${training_time}, testing_time=${testing_time}, accuracy=${accuracy}"
    echo "-------------------------------------------------------"
done

echo "All configurations completed. Summary stored in ${overall_results}."

#!/bin/bash
# run_svm.sh
# This script iterates over different dataset fractions and runs SVM.py for each fraction.
# SVM.py must accept two command-line arguments: kernel type and frac.
# It prints a single number (the training latency) and saves the model into a pickle file.
# The overall summary is saved in SVM_testing_results.txt.

# Define the dataset fractions to test.
frac=(0.001 0.002 0.004 0.008)

# Overall results file for summary.
overall_results="SVM_testing_results.txt"
> "$overall_results"
echo -e "frac\ttraining_latency" >> "$overall_results"

# Iterate over each fraction.
for (( i=0; i<${#frac[@]}; i++ )); do
    frac_val=${frac[$i]}
    echo "Running SVM with kernel rbf and frac=${frac_val}"
    
    # Run SVM.py with kernel type 'rbf' and the current fraction.
    training_latency=$(python3 SVM.py "rbf" "$frac_val")
    
    # Append the fraction and training latency to the results file.
    echo -e "${frac_val}\t${training_latency}" >> "$overall_results"
    
    echo "Completed: training_latency = ${training_latency} seconds for frac=${frac_val}"
    echo "-------------------------------------------------------"
done

echo "All configurations completed. Summary stored in ${overall_results}."

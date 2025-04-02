#!/bin/bash
# run_svm.sh
# This script iterates over different dataset fractions and cache sizes,
# and runs SVM.py for each combination.
# SVM.py must accept three command-line arguments: kernel type, frac, and cache_size,
# and it must print a single number (the training latency) while saving the model into a pickle file.
# The overall summary is saved in SVM_testing_results.txt.

# Define the dataset fractions to test.
frac=(0.001 0.002 0.004 0.008 0.016 0.032 0.05)

# Define the cache sizes (in MB).
cache_sizes=(200 400 800 1600 3200 6400 12800 25600)

# Overall results file for summary.
overall_results="SVM_testing_results.txt"
> "$overall_results"
echo -e "frac\tcache_size\ttraining_latency" >> "$overall_results"

# Outer loop: iterate over each fraction.
for (( i=0; i<${#frac[@]}; i++ )); do
    frac_val=${frac[$i]}
    # Inner loop: iterate over each cache size.
    for (( j=0; j<${#cache_sizes[@]}; j++ )); do
        cache_val=${cache_sizes[$j]}
        echo "Running SVM with kernel rbf, frac=${frac_val} and cache_size=${cache_val}MB"
        
        # Run SVM.py with kernel type 'rbf', the current fraction, and cache size.
        training_latency=$(python3 SVM.py "rbf" "$frac_val" "$cache_val")
        
        # Append the result to the overall results file.
        echo -e "${frac_val}\t${cache_val}\t${training_latency}" >> "$overall_results"
        
        echo "Completed: training_latency = ${training_latency} seconds for frac=${frac_val} and cache_size=${cache_val}MB"
        echo "-------------------------------------------------------"
    done
done

echo "All configurations completed. Summary stored in ${overall_results}."

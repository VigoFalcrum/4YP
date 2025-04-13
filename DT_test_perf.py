#!/usr/bin/env python3
from sklearn.tree import DecisionTreeClassifier
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import pickle
import sklearn

def preprocess_data():
    # Parse command-line arguments for classifier parameters.
    # This branch expects sys.argv to be: [script, max_depth, min_samples_split]
    if len(sys.argv) < 3:
        print("Usage: python3 DT_train_perf.py <max_depth> <min_samples_split>")
        sys.exit(1)
    arg_max_depth = sys.argv[1]
    arg_min_samples_split = sys.argv[2]
    max_depth_param = int(arg_max_depth)
    min_split_param = int(arg_min_samples_split)

    # Data preprocessing:
    datatypes = {'srcip': int, 'sport': int, 'dstip': int, 'dsport': int, 'proto': int, 'state': int, 'dur': float,
                 'sbytes': int, 'dbytes': int, 'sttl': int, 'dttl': int, 'sloss': int, 'dloss': int, 'service': int,
                 'Sload': float, 'Dload': float, 'Spkts': int, 'Dpkts': int, 'swin': int, 'dwin': int, 'stcpb': int,
                 'dtcpb': int, 'smeansz': int, 'dmeansz': int, 'trans_depth': int, 'res_bdy_len': int, 'Sjit': float,
                 'Djit': float, 'Sintpkt': float, 'Dintpkt': float, 'tcprtt': float,
                 'synack': float, 'ackdat': float, 'is_sm_ips_ports': int, 'ct_state_ttl': int, 'ct_flw_http_mthd': int,
                 'ct_srv_src': int, 'ct_srv_dst': int, 'ct_dst_ltm': int,
                 'ct_src_ltm': int, 'ct_src_dport_ltm': int, 'ct_dst_sport_ltm': int, 'ct_dst_src_ltm': int}

    col_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss',
                 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
                 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack',
                 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src',
                 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

    # Train data
    X_train_data = 'X_train.parquet'
    y_train_data = 'y_train.parquet'
    X_train = pd.read_parquet(X_train_data)
    y_train = pd.read_parquet(y_train_data)
    X_train = X_train.astype(datatypes)
    X_train.columns = col_names
    y_train.columns = ['label']
    X_train = X_train.drop(X_train.index[0])
    y_train = y_train.drop(y_train.index[0])

    # Test data (if needed)
    X_test_data = 'X_test.parquet'
    y_test_data = 'y_test.parquet'
    X_test = pd.read_parquet(X_test_data)
    y_test = pd.read_parquet(y_test_data)
    X_test = X_test.astype(datatypes)
    X_test.columns = col_names
    y_test.columns = ['label']
    X_test = X_test.drop(X_test.index[0])
    y_test = y_test.drop(y_test.index[0])

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    # Save preprocessed training data for the training subprocess
    X_test.to_parquet("X_test_processed.parquet", index=False)
    y_test.to_frame().to_parquet("y_test_processed.parquet", index=False)
    print("✅ Preprocessing complete and data saved.")

def test_classifier():
    # This branch expects sys.argv to be: [script, "--train", max_depth, min_samples_split]
    if len(sys.argv) < 4:
        print("Usage: python3 DT_train_perf.py --train <max_depth> <min_samples_split>")
        sys.exit(1)
    # Skip the flag at index 1
    arg_max_depth = sys.argv[2]
    arg_min_samples_split = sys.argv[3]
    max_depth_param = int(arg_max_depth)
    min_split_param = int(arg_min_samples_split)

    # Load the preprocessed data
    X_test = pd.read_parquet("X_test_processed.parquet")
    y_test = pd.read_parquet("y_test_processed.parquet")
    y_test = y_test.values.ravel()

    # Draw 100 random samples (as a DataFrame) from X_test
    random_samples = X_test.sample(n=100, random_state=29)

    # Load the trained classifier
    filename = f"DT_{max_depth_param}.pkl"
    with open(filename, 'rb') as f:
        clf = pickle.load(f)

    start_time = time.perf_counter()
    # Measuring inference time
    for i in range(100):
        # Select a single sample as a DataFrame (preserving column names)
        sample = random_samples.iloc[[i]]
        clf.predict(sample)
    end_time = time.perf_counter()
    inference_time = end_time - start_time

    # Estimating the Python loop overhead
    for i in range(100):
        # Select a single sample as a DataFrame (preserving column names)
        sample = random_samples.iloc[[i]]
    end_time = time.perf_counter()
    print((inference_time - end_time + start_time)/100)

if __name__ == '__main__':
    if "--test" in sys.argv:
        test_classifier()
    else:
        # Main process: parse parameters and run preprocessing.
        if len(sys.argv) < 3:
            print("Usage: python3 DT_test_perf.py <max_depth> <min_samples_split>")
            sys.exit(1)
        preprocess_data()
        # Launch the testing subprocess with additional arguments.
        subprocess.run([sys.executable, sys.argv[0], "--test", sys.argv[1], sys.argv[2]])

#!/usr/bin/env python3
from sklearn.tree import DecisionTreeClassifier
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import pickle
import sklearn

# Parse command-line arguments for classifier parameters.
if len(sys.argv) > 2:
    arg_max_depth = sys.argv[1]
    arg_min_samples_split = sys.argv[2]

    max_depth_param = int(arg_max_depth)
    min_split_param = int(arg_min_samples_split)

def preprocess_data():
    # Data preprocessing:
    datatypes = {'srcip' : int, 'sport' : int, 'dstip' : int, 'dsport' : int, 'proto' : int, 'state' : int, 'dur' : float,
                 'sbytes' : int, 'dbytes' : int, 'sttl' : int, 'dttl' : int, 'sloss' : int, 'dloss' : int, 'service' : int,
                 'Sload' : float, 'Dload' : float, 'Spkts' : int, 'Dpkts' : int, 'swin' : int, 'dwin' : int, 'stcpb' : int,
                 'dtcpb' : int, 'smeansz' : int, 'dmeansz' : int, 'trans_depth' : int, 'res_bdy_len' : int, 'Sjit' : float,
                 'Djit' : float, 'Sintpkt' : float, 'Dintpkt' : float, 'tcprtt' : float,
                 'synack' : float, 'ackdat' : float, 'is_sm_ips_ports' : int, 'ct_state_ttl' : int, 'ct_flw_http_mthd' : int,
                 'ct_srv_src' : int, 'ct_srv_dst' : int, 'ct_dst_ltm' : int,
                 'ct_src_ltm' : int, 'ct_src_dport_ltm' : int, 'ct_dst_sport_ltm' : int, 'ct_dst_src_ltm' : int}

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
    
    # Test data (if needed elsewhere; here we focus on training)
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
    
    # Save preprocessed training data for the subprocess
    X_train.to_parquet("X_train_processed.parquet")
    y_train.to_frame().to_parquet("y_train_processed.parquet")
    print("✅ Preprocessing complete and data saved.")

def train_classifier():
    # Import only what is needed here so that this code runs in the training-only subprocess.
    import time
    import pandas as pd
    
    # Load the preprocessed data
    X_train = pd.read_parquet("X_train_processed.parquet")
    y_train = pd.read_parquet("y_train_processed.parquet")
    y_train = y_train.values.ravel()
    
    # Create the classifier

    clf = DecisionTreeClassifier(max_depth=max_depth_param, min_samples_split=min_samples_split_param)
    
    start_time = time.perf_counter()
    tree.fit(X_train, y_train)
    end_time = time.perf_counter()
    print(end_time - start_time)

    filename = f"DT_{max_depth_param}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)

if __name__ == '__main__':
    # If the "--train" flag is provided, only run the training section.
    if "--train" in sys.argv:
        train_classifier()
    else:
        preprocess_data()
        # Launch the training part in a subprocess.
        print("Launching training subprocess...")
        subprocess.run([sys.executable, sys.argv[0], "--train"])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
from sklearn.metrics import accuracy_score
import sys

# Data preprocessing
datatypes = {'srcip' : int, 'sport' : int, 'dstip' : int, 'dsport' : int, 'proto' : int, 'state' : int, 'dur' : float, \
             'sbytes' : int, 'dbytes' : int, 'sttl' : int, 'dttl' : int, 'sloss' : int, 'dloss' : int, 'service' : int, \
             'Sload' : float, 'Dload' : float, 'Spkts' : int, 'Dpkts' : int, 'swin' : int, 'dwin' : int, 'stcpb' : int, \
             'dtcpb' : int, 'smeansz' : int, 'dmeansz' : int, 'trans_depth' : int, 'res_bdy_len' : int, 'Sjit' : float, \
             'Djit' : float, 'Sintpkt' : float, 'Dintpkt' : float, 'tcprtt' : float, \
             'synack' : float, 'ackdat' : float, 'is_sm_ips_ports' : int, 'ct_state_ttl' : int, 'ct_flw_http_mthd' : int, \
             'ct_srv_src' : int, 'ct_srv_dst' : int, 'ct_dst_ltm' : int, \
             'ct_src_ltm' : int, 'ct_src_dport_ltm' : int, 'ct_dst_sport_ltm' : int, 'ct_dst_src_ltm' : int}

col_names = ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', \
             'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', \
             'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', \
             'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', \
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

# Test data
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

# FINISHED DATASET PREPROCESSING

n_estimators=[5, 5, 5, 15, 15, 15, 45, 45, 45, 90, 90, 90, 140, 140, 140, 200, 200, 200, 270, 270, 270]
max_depths=[7, 15, 25, 7, 15, 25, 7, 15, 25, 7, 15, 25, 7, 15, 25, 7, 15, 25, 7, 15, 25]

for i in range(len(n_estimators)):
    filename = f"RF_{n_estimators[i]}_{max_depths[i]}.pkl"
    with open(filename, 'rb') as file:
        clf = pickle.load(file)
    
    y_pred = clf.predict(X_test)
    # Ensure y_test is of integer type
    y_test_int = y_test.astype('int64')
    
    print(f"RF model with {n_estimators[i]} n_estimators and {max_depths[i]} max_depth")
    print("Model accuracy score with criterion entropy: {0:0.4f}".format(
          accuracy_score(y_test_int, y_pred)))
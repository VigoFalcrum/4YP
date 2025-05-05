import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time
import subprocess
import pandas as pd
import numpy as np
import pickle
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler
import random

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, num_classes):
        super(DeepNN, self).__init__()
        layers = []
        
        # First layer: from input to hidden_size
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        # Hidden layers: hidden_size -> hidden_size
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            
        # Final layer: from hidden_size to num_classes
        layers.append(nn.Linear(hidden_size, num_classes))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

def preprocess_data():
    # Parse command-line arguments for classifier parameters.
    # This branch expects sys.argv to be: [script, n_estimators, max_depth, min_samples_split, n_jobs]
    arg_layer_width = sys.argv[2]
    arg_depth = sys.argv[1]
    arg_cores = sys.argv[3]

    layer_width = int(arg_layer_width)
    nn_depth = int(arg_depth)
    cores = int(arg_cores)

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
    X_train.to_parquet("X_train_processed.parquet", index=False)
    y_train.to_frame().to_parquet("y_train_processed.parquet", index=False)
    print("✅ Preprocessing complete and data saved.")

def train_classifier():
    
    arg_layer_width = sys.argv[3]
    arg_depth = sys.argv[2]
    arg_cores = sys.argv[4]

    layer_width = int(arg_layer_width)
    nn_depth = int(arg_depth)
    cores = int(arg_cores)

    torch.set_num_threads(cores)
    torch.manual_seed(29)
    np.random.seed(29)
    random.seed(29)

    X_train = pd.read_parquet("X_train_processed.parquet")
    y_train = pd.read_parquet("y_train_processed.parquet")
    y_train = y_train.values.ravel()

    device = 'cpu'
    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)

    input_size, hidden_size, num_classes = X_train.shape[1], layer_width, 2
    
    model = DeepNN(input_size, hidden_size, nn_depth, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    start_time = time.perf_counter()

    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = time.perf_counter()

    # Save the trained classifier
    filename = f"NN_{nn_depth}_{layer_width}.pth"
    torch.save(model.state_dict(), filename)
    joblib.dump(scaler, f"scaler_{nn_depth}_{layer_width}.pkl")

    print(f"Training time: {end_time - start_time:.5f} seconds")

if __name__ == '__main__':
    if "--train" in sys.argv:
        train_classifier()
    else:
        # Main process: parse parameters and run preprocessing.
        if len(sys.argv) < 4:
            print("Usage: python3 NN_train_perf.py <layer_width> <depth> <cores>")
            sys.exit(1)
        preprocess_data()
        # Launch the training subprocess with additional arguments.
        subprocess.run([sys.executable, sys.argv[0], "--train", sys.argv[1], sys.argv[2], sys.argv[3]])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import random
import time
import sys

# Set random seeds for reproducibility
torch.manual_seed(29)
np.random.seed(29)
random.seed(29)
torch.cuda.empty_cache()

# Parse command-line arguments for classifier parameters.
if len(sys.argv) > 2:
    arg_layer_width = sys.argv[1]
    arg_depth = sys.argv[2]
   # arg_cores = sys.argv[3]

    layer_width = int(arg_layer_width)
    nn_depth = int(arg_depth)
   # cores = int(arg_cores)
else:
    # Default values if no arguments are provided.
    print("Yo brotha\n")
# layer_width = 8
# nn_depth = 4

# Limit the CPU threadcount
# torch.set_num_threads(cores)

# Check for GPU availability
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'
# print(f'Using device: {device}')

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
X_train_data = 'X_train_GPU.parquet'
y_train_data = 'y_train_GPU.parquet'
X_train = pd.read_parquet(X_train_data)
y_train = pd.read_parquet(y_train_data)
X_train = X_train.astype(datatypes)

X_train.columns = col_names
y_train.columns = ['label']
X_train = X_train.drop(X_train.index[0])
y_train = y_train.drop(y_train.index[0])

y_train = y_train.squeeze()

scaler = StandardScaler()

X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=0.1, stratify=y_train, random_state=29)

X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)

# Define the Neural Network

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

# Model initialization and device allocation
input_size, hidden_size, num_classes = X_train.shape[1], layer_width, 2
model = DeepNN(input_size, hidden_size, nn_depth, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Training loop
num_epochs = 10
torch.cuda.synchronize()
start_time = time.perf_counter()
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
end_time = time.perf_counter()
   # print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

filename = f"NN_{nn_depth}_{layer_width}.pth"
torch.save(model.state_dict(), filename)

print((end_time - start_time)/num_epochs)

#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
import math

# ──────── Argument parsing ────────
# Usage: infer.py <nn_depth> <hidden_size>
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <nn_depth> <hidden_size>", file=sys.stderr)
    sys.exit(1)

nn_depth   = int(sys.argv[1])
hidden_size = int(sys.argv[2])

model_pth  = f"NN_{nn_depth}_{hidden_size}.pth"
scaler_pkl = f"scaler_{nn_depth}_{hidden_size}.pkl"

# ──────── Hyperparameters ────────
batch_size = 32_000

# ──────── Seed & Device ────────
torch.manual_seed(29)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ──────── Load & Preprocess Data ────────
# 1) Read parquet
X = pd.read_parquet("X_test.parquet")
y = pd.read_parquet("y_test.parquet")

# 2) Drop the header‑row quirk
X = X.drop(X.index[0]).astype(int)
y = y.drop(y.index[0]).squeeze().astype(int)

# 3) Rename columns (must match training)
X.columns = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss',
    'dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
    'trans_depth','res_bdy_len','Sjit','Djit','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm'
]

# 4) Load your pretrained scaler and transform
scaler = joblib.load(scaler_pkl)
X_np  = scaler.transform(X)       # NumPy array
y_np  = y.values                  # 1‑D NumPy array

# ──────── Move to Device & Batchify ────────
X_dev = torch.from_numpy(X_np).float().to(device, non_blocking=True)
y_dev = torch.from_numpy(y_np).long().to(device, non_blocking=True)

num_samples = X_dev.size(0)
batches     = [
    (X_dev[i : min(i+batch_size, num_samples)],
     y_dev[i : min(i+batch_size, num_samples)])
    for i in range(0, num_samples, batch_size)
]

# ──────── Model Definition ────────
class DeepNN(nn.Module):
    def __init__(self, in_sz, hid_sz, depth, num_cls=2):
        super().__init__()
        layers = [nn.Linear(in_sz, hid_sz), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hid_sz, hid_sz), nn.ReLU()]
        layers.append(nn.Linear(hid_sz, num_cls))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = DeepNN(X_dev.shape[1], hidden_size, nn_depth).to(device)
model.load_state_dict(torch.load(model_pth, map_location=device))
model.eval()

# ──────── Warm-up ────────
for _ in range(5):
    _ = model(batches[0][0])

# ──────── Inference ────────
torch.cuda.synchronize()
start_time = time.perf_counter()
with torch.no_grad():
    for bx, by in batches:
        logits = model(bx)

torch.cuda.synchronize()
end_time = time.perf_counter()

n_batches = math.ceil(len(y_np)/batch_size)
print((end_time - start_time))

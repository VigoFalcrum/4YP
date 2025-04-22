#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import time

# ─── Argument parsing
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <nn_depth> <hidden_size>", file=sys.stderr)
    sys.exit(1)
nn_depth    = int(sys.argv[1])
hidden_size = int(sys.argv[2])

model_pth   = f"NN_{nn_depth}_{hidden_size}.pth"
scaler_pkl  = f"scaler_{nn_depth}_{hidden_size}.pkl"

# ─── Device & seed
torch.manual_seed(29)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─── Load data
X = pd.read_parquet("X_test.parquet").drop(0).astype(int)
# rename columns as before…

# ─── Sample and scale
rng = np.random.RandomState(29)
sample_df = X.sample(n=100, random_state=rng).reset_index(drop=True)

scaler    = joblib.load(scaler_pkl)
sample_np = scaler.transform(sample_df)      # shape: (100, D)

# ─── Convert to tensor once
sample_tensors = [
    torch.from_numpy(sample_np[i : i+1]).float().to(device)
    for i in range(sample_np.shape[0])
]

# ─── Model definition (same as before)
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

model = DeepNN(sample_np.shape[1], hidden_size, nn_depth).to(device)
model.load_state_dict(torch.load(model_pth, map_location=device))
model.eval()

# ─── Warm‑up (single‑sample path)
with torch.no_grad():
    for _ in range(10):
        _ = model(sample_tensors[0])
    torch.cuda.synchronize()

# ─── Measure per‑sample latency
times = []
torch.cuda.synchronize()      # wait for any prior work
t0 = time.perf_counter()
with torch.no_grad():
    for tensor in sample_tensors:
        _ = model(tensor)
torch.cuda.synchronize()      # wait until this inference is done
t1 = time.perf_counter()

print(t1 - t0)

#!/usr/bin/env python3
import sys
import time
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler

# ──────── Argument parsing ────────
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <layer_width> <depth>", file=sys.stderr)
    sys.exit(1)

layer_width = int(sys.argv[1])
nn_depth    = int(sys.argv[2])

# ──────── Seed/GPU setup ────────
torch.manual_seed(29)
np.random.seed(29)
random.seed(29)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

# ──────── Preprocessing ────────
try:
    # 1) Read parquet
    X = pd.read_parquet("X_train_GPU.parquet")
    y = pd.read_parquet("y_train_GPU.parquet")

    # 2) Drop the header‑row quirk
    X = X.drop(X.index[0]).astype(int)  
    y = y.drop(y.index[0]).squeeze().astype(int)

    # 3) Rename columns
    X.columns = [
        'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss',
        'dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
        'trans_depth','res_bdy_len','Sjit','Djit','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
        'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
        'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm'
    ]

    # 4) Scale
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X)        # NumPy array on CPU
    y_np = y.values                       # 1‑D NumPy array

except Exception as e:
    print("❌ Preprocessing failed:", e, file=sys.stderr)
    raise

# ──────── Move entire dataset to GPU ────────
X_gpu = torch.from_numpy(X_np).float().to(device, non_blocking=True)
y_gpu = torch.from_numpy(y_np).long().to(device, non_blocking=True)

# ──────── Pre‑slice into on‑device batches ────────
n_batches  = 10
batch_size = 32_000
batches = []
for i in range(n_batches):
    start = i * batch_size
    end   = start + batch_size
    batches.append((X_gpu[start:end], y_gpu[start:end]))

# ──────── Model definition ────────
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

model     = DeepNN(X_gpu.shape[1], layer_width, nn_depth).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ──────── Warm‑up ────────
print("Warmup batch →", file=sys.stderr)
model.train()
bx, by = batches[0]
out = model(bx)
loss = criterion(out, by)
optimizer.zero_grad()
loss.backward()
optimizer.step()

# ──────── Timed runs ────────
trials = 5
times  = []

for t in range(trials):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for bx, by in batches:
        optimizer.zero_grad()
        out  = model(bx)
        loss = criterion(out, by)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"Run {t+1}/{trials}: {elapsed:.4f}s", file=sys.stderr)
    times.append(elapsed)

times = np.array(times)
print(f"Mean latency: {times.mean():.4f}s ± {times.std():.4f}s over {trials} runs")

# ──────── Save weights ────────
torch.save(model.state_dict(), f"NN_{nn_depth}_{layer_width}.pth")

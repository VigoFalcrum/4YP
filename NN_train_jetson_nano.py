#!/usr/bin/env python3
import sys, time, random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
    X = X.drop(X.index[0]).astype(int)  # you can cast floats if needed
    y = y.drop(y.index[0]).squeeze().astype(int)

    # 3) Rename/remove columns as before
    X.columns = [ /* your col_names list here */ ]
    # y is just a Series of ints

    # 4) Scale
    scaler = StandardScaler()
    X_np = scaler.fit_transform(X)        # still a NumPy array
    y_np = y.values                       # 1‑D NumPy array

    # 5) Tensor‑ify and move to GPU
    X_tensor = torch.from_numpy(X_np).float().to(device)
    y_tensor = torch.from_numpy(y_np).long().to(device)

except Exception as e:
    print("❌ Preprocessing failed:", e, file=sys.stderr)
    raise

# ──────── DataLoader ────────
dataset = TensorDataset(X_tensor, y_tensor)
loader  = DataLoader(
    dataset,
    batch_size=32_000,
    shuffle=False,
    pin_memory=True,
    num_workers=4,
    persistent_workers=True
)

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

model     = DeepNN(X_tensor.shape[1], layer_width, nn_depth).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# ──────── Warm‑up ────────
print("Warmup batch →", file=sys.stderr)
model.train()
data, labels = next(iter(loader))
data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
out = model(data)
loss = criterion(out, labels)
optimizer.zero_grad(); loss.backward(); optimizer.step()

# ──────── Timing ────────
trials = 5
times  = []
for i in range(trials):
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        out = model(batch_X)
        loss = criterion(out, batch_y)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed = t1 - t0
    print(f"Run {i+1}/{trials}: {elapsed:.4f}s", file=sys.stderr)
    times.append(elapsed)

times = np.array(times)
print(f"Mean latency: {times.mean():.4f}s ± {times.std():.4f}s over {trials} runs")

# ──────── Save weights ────────
torch.save(model.state_dict(), f"NN_{nn_depth}_{layer_width}.pth")

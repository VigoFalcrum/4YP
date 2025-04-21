import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.preprocessing import StandardScaler

# ──────── Argument parsing ────────
# Usage: infer.py <nn_depth> <hidden_size>
if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <nn_depth> <hidden_size>", file=sys.stderr)
    sys.exit(1)

nn_depth   = sys.argv[1]
hidden_size  = sys.argv[2]

model_pth = f"NN_{nn_depth}_{hidden_size}.pth"
scaler_pkl = f"scaler_{nn_depth}_{hidden_size}.pkl"

# optional batch size
batch_size = 32_000

# ──────── Seed/GPU setup ────────
torch.manual_seed(29)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

# ──────── Preprocessing ────────
# 1) Read parquet
X = pd.read_parquet("X_test.parquet")
y = pd.read_parquet("y_test.parquet")

# 2) Drop the header‑row quirk
X = X.drop(X.index[0]).astype(int)
y = y.drop(y.index[0]).squeeze().astype(int)

# 3) Rename columns (same as training)
X.columns = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes','sttl','dttl','sloss',
    'dloss','service','Sload','Dload','Spkts','Dpkts','swin','dwin','stcpb','dtcpb','smeansz','dmeansz',
    'trans_depth','res_bdy_len','Sjit','Djit','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','ct_srv_src','ct_srv_dst','ct_dst_ltm',
    'ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm'
]

# 4) Load pre‑fitted scaler and transform
scaler: StandardScaler = joblib.load(scaler_pkl)
X_np = scaler.transform(X)        # NumPy array on CPU
y_np = y.values                   # 1‑D NumPy array

# ──────── Move entire dataset to device ────────
X_dev = torch.from_numpy(X_np).float().to(device, non_blocking=True)
y_dev = torch.from_numpy(y_np).long().to(device, non_blocking=True)

# ──────── Pre‑slice into batches ────────
num_samples = X_dev.size(0)
batches = []
for start in range(0, num_samples, batch_size):
    end = min(start + batch_size, num_samples)
    batches.append((X_dev[start:end], y_dev[start:end]))

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

model = DeepNN(X_dev.shape[1], hidden_size, nn_depth).to(device)
model.load_state_dict(torch.load(model_pth, map_location=device))
model.eval()

# ──────── Inference ────────
all_preds = []
all_labels = []

with torch.no_grad():
    for bx, by in batches:
        logits = model(bx)
        preds  = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(by.cpu().numpy())

all_preds  = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

accuracy = (all_preds == all_labels).mean()
print(f"Inference on {num_samples} samples → Accuracy: {accuracy*100:.2f}%")

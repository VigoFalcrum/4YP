#!/usr/bin/env python3
import sys, time, random, pathlib
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

# ───────────────────────────────────────────
# Model
# ───────────────────────────────────────────
class DeepNN(nn.Module):
    def __init__(self, in_dim, hidden, depth, out_dim=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, out_dim))
        # match training script
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ───────────────────────────────────────────
# Schema & helpers
# ───────────────────────────────────────────
_DATATYPES = {
    'srcip': int, 'sport': int, 'dstip': int, 'dsport': int,
    'proto': int, 'state': int, 'dur': float, 'sbytes': int,
    'dbytes': int, 'sttl': int, 'dttl': int, 'sloss': int,
    'dloss': int, 'service': int, 'Sload': float, 'Dload': float,
    'Spkts': int, 'Dpkts': int, 'swin': int, 'dwin': int,
    'stcpb': int, 'dtcpb': int, 'smeansz': int, 'dmeansz': int,
    'trans_depth': int, 'res_bdy_len': int, 'Sjit': float, 'Djit': float,
    'Sintpkt': float, 'Dintpkt': float, 'tcprtt': float, 'synack': float,
    'ackdat': float, 'is_sm_ips_ports': int, 'ct_state_ttl': int,
    'ct_flw_http_mthd': int, 'ct_srv_src': int, 'ct_srv_dst': int,
    'ct_dst_ltm': int, 'ct_src_ltm': int, 'ct_src_dport_ltm': int,
    'ct_dst_sport_ltm': int, 'ct_dst_src_ltm': int
}
_COLS = list(_DATATYPES.keys())
device = "cpu"

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(_DATATYPES)
    df.columns = _COLS
    return df.iloc[1:].reset_index(drop=True)  # drop the duplicated header row

def preprocess_test_data():
    """Force‐recreate the processed test files from raw parquet."""
    X = _clean(pd.read_parquet("X_test.parquet"))
    y = pd.read_parquet("y_test.parquet").iloc[1:, 0].reset_index(drop=True)
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=32_000, stratify=y, random_state=29)
    X_sub.to_parquet("X_test_batch.parquet", index=False)
    y_sub.to_frame().to_parquet("y_test_batch.parquet", index=False)

# ───────────────────────────────────────────
# Inference
# ───────────────────────────────────────────
def run_inference(layer_width: int, depth: int, cores: int, skip_preprocess: bool):
    torch.set_num_threads(cores)
    torch.manual_seed(29); random.seed(29); np.random.seed(29)

    if not skip_preprocess:
        preprocess_test_data()

    # Load processed test set
    X_test = pd.read_parquet("X_test_batch.parquet")

    # Restore scaler & model
    scaler = joblib.load(f"scaler_{depth}_{layer_width}.pkl")
    model  = DeepNN(X_test.shape[1], layer_width, depth).to(device)
    state  = torch.load(f"NN_{depth}_{layer_width}.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Prepare tensors
    X_t = torch.tensor(scaler.transform(X_test), dtype=torch.float32).to(device)
    
    # Inference
    t0 = time.perf_counter()
    for _ in range(5):
        with torch.no_grad():
            logits = model(X_t)
    t1 = time.perf_counter()

    total = (t1 - t0)/5
    print(f"Training time: {total:.5f} seconds")
# ───────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--test":
        # CLI: python NN_infer.py --test <width> <depth> <cores>
        if len(args) != 4:
            print("Usage: python NN_infer.py --test <layer_width> <depth> <cores>")
            sys.exit(1)
        _, w, d, c = args
        run_inference(int(w), int(d), int(c), skip_preprocess=True)

    else:
        # CLI: python NN_infer.py <width> <depth> <cores>
        if len(args) != 3:
            print("Usage:")
            print("  python NN_infer.py <layer_width> <depth> <cores>")
            print("  python NN_infer.py --test <layer_width> <depth> <cores>")
            sys.exit(1)
        w, d, c = args
        run_inference(int(w), int(d), int(c), skip_preprocess=False)

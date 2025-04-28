#!/usr/bin/env python3
# run_v70_infer.py  (datatype-aware)

import time, pickle, numpy as np, pandas as pd, xir, vart

# ---------- USER SETTINGS ----------
X_SOURCE   = "unsw_nb15.parquet"            # or "unsw_X.npy"
LABEL_COL  = "label"
BATCH      = 32
MODEL_PATH = "DeepNN_pt/DeepNN_pt.xmodel"
SCALER_PKL = "scaler.pkl"

# explicit feature dtypes
datatypes = {
    'srcip': int, 'sport': int, 'dstip': int, 'dsport': int, 'proto': int,
    'state': int, 'dur': float, 'sbytes': int, 'dbytes': int, 'sttl': int,
    'dttl': int, 'sloss': int, 'dloss': int, 'service': int, 'Sload': float,
    'Dload': float, 'Spkts': int, 'Dpkts': int, 'swin': int, 'dwin': int,
    'stcpb': int, 'dtcpb': int, 'smeansz': int, 'dmeansz': int,
    'trans_depth': int, 'res_bdy_len': int, 'Sjit': float, 'Djit': float,
    'Sintpkt': float, 'Dintpkt': float, 'tcprtt': float, 'synack': float,
    'ackdat': float, 'is_sm_ips_ports': int, 'ct_state_ttl': int,
    'ct_flw_http_mthd': int, 'ct_srv_src': int, 'ct_srv_dst': int,
    'ct_dst_ltm': int, 'ct_src_ltm': int, 'ct_src_dport_ltm': int,
    'ct_dst_sport_ltm': int, 'ct_dst_src_ltm': int
}
feature_cols = list(datatypes.keys())  # preserves order
# ------------------------------------

# ----------  SETUP DPU RUNNER ----------
graph = xir.Graph.deserialize(MODEL_PATH)
dpu_sg = next(s for s in graph.get_root_subgraph().children()
              if s.get_attr("device") == "DPU")
runner = vart.Runner.create_runner(dpu_sg, "run")

in_t  = runner.get_input_tensors()[0]
out_t = runner.get_output_tensors()[0]
feat_dim  = in_t.dims[1]
n_classes = out_t.dims[1]

assert feat_dim == len(feature_cols), \
    f"DPU expects {feat_dim} features but dict has {len(feature_cols)}"

scaler = pickle.load(open(SCALER_PKL, "rb"))

def send_to_dpu(batch_f32: np.ndarray) -> np.ndarray:
    batch_f32 = np.ascontiguousarray(batch_f32.astype("float32"))
    out = np.empty((len(batch_f32), n_classes), dtype="float32")
    jid = runner.execute_async([batch_f32], [out])
    runner.wait(jid)
    return out

# ----------  DATA STREAMER ----------
def stream_batches():
    if X_SOURCE.endswith(".npy"):                       # ---------- NumPy
        X_all = np.load(X_SOURCE, mmap_mode='r')
        for i in range(0, len(X_all), BATCH):
            yield X_all[i:i+BATCH], None
    else:                                               # ---------- Parquet
        cols = feature_cols + ([LABEL_COL] if LABEL_COL else [])
        for chunk in pd.read_parquet(X_SOURCE, columns=cols, chunksize=BATCH):
            if LABEL_COL in chunk:
                y = chunk.pop(LABEL_COL).to_numpy()
            else:
                y = None
            # enforce exact dtypes, then to float32 NumPy
            chunk = chunk.astype(datatypes)
            yield chunk.to_numpy(np.float32), y

# ----------  INFERENCE ----------
n_seen, t_total = 0, 0.0
for raw, _ in stream_batches():
    t0 = time.time()
    batch = scaler.transform(raw).astype("float32")
    logits = send_to_dpu(batch)
    preds  = logits.argmax(axis=1)
    dt = time.time() - t0

    n_seen += len(raw);  t_total += dt
    print(f"[{n_seen:>8}] batch={len(raw):3}  "
          f"lat={dt*1000:6.2f} ms  pred0={preds[0]}")

print(f"\nProcessed {n_seen} rows — "
      f"avg {t_total/n_seen*1000:.2f} ms,  "
      f"{n_seen/t_total:.1f} rows/s")

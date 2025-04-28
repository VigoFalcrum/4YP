import time, numpy as np, xir, vart, os

# ---------------- USER CONFIG -----------------
X_INT8_NPY = "unsw_X_int8.npy"          # INT8 features (N, feat)
Y_NPY      = "unsw_y.npy"               # optional labels
MODEL_PATH = "DeepNN_pt.xmodel"
BATCH      = 300_000                        # try 128..512 for peak QPS
# ----------------------------------------------

# ---- 1.  create DPU runner -------------------
graph  = xir.Graph.deserialize(MODEL_PATH)
dpu_sg = next(sg for sg in graph.get_root_subgraph().get_children()
              if sg.get_attr("device") == "DPU")
runner = vart.Runner.create_runner(dpu_sg, "run")

in_tensor, out_tensor = runner.get_input_tensors()[0], runner.get_output_tensors()[0]
feat_dim  = in_tensor.dims[1]
n_classes = out_tensor.dims[1]

# ---- 2.  mmap INT8 dataset -------------------
X_all = np.load(X_INT8_NPY, mmap_mode="r")        # shape (N, feat) int8
assert X_all.shape[1] == feat_dim, "feature-count mismatch!"

have_y = os.path.exists(Y_NPY)
if have_y:
    y_all = np.load(Y_NPY, mmap_mode="r")

def dpu(batch_int8: np.ndarray) -> np.ndarray:
    """Send one INT8 batch to DPU and return float32 logits."""
    batch_int8 = np.ascontiguousarray(batch_int8.astype("int8"))
    out = np.empty((len(batch_int8), n_classes), dtype="float32")
    jid = runner.execute_async([batch_int8], [out]); ret=runner.wait(jid)
    print(ret)
    return out

# ---- 3.  streaming inference -----------------
# Warm-up
_ = dpu(X_all[0:BATCH])

start_time = time.perf_counter()
for idx in range(0, len(X_all), BATCH):
    batch_int8 = X_all[idx: idx + BATCH]
    preds = dpu(batch_int8).argmax(1)

end_time = time.perf_counter()
latency = end_time - start_time

start_time = time.perf_counter()
for idx in range(0, len(X_all), BATCH):
    batch_int8 = X_all[idx: idx + BATCH]

end_time = time.perf_counter()

print(latency - end_time + start_time)


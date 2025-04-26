# quantize_ptq_tabular.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import importlib.util # To load model definition dynamically

# Import the Vitis AI Quantizer API
from pytorch_nndct.apis import torch_quantizer

# --- Configuration ---
# Input shape for a single sample (features,)
# Batch dimension is handled by DataLoader
INPUT_SHAPE = (43,)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Helper Functions ---

def load_model_from_file(model_def_path, model_weights_path, model_class_name,
                         input_size, hidden_size, num_hidden_layers, num_classes):
    """Loads PyTorch model definition and weights."""
    print(f"Loading model definition '{model_class_name}' from: {model_def_path}")
    print(f"Loading float weights from: {model_weights_path}")

    if not os.path.exists(model_def_path):
        raise FileNotFoundError(f"Model definition file not found: {model_def_path}")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found: {model_weights_path}")

    try:
        spec = importlib.util.spec_from_file_location("model_module", model_def_path)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)
        ModelClass = getattr(model_module, model_class_name)
    except Exception as e:
        raise ImportError(f"Could not load model class '{model_class_name}' from {model_def_path}: {e}")

    # Instantiate the model with provided hyperparameters
    float_model = ModelClass(input_size=input_size,
                             hidden_size=hidden_size,
                             num_hidden_layers=num_hidden_layers,
                             num_classes=num_classes)

    # Load the state dictionary
    float_model.load_state_dict(torch.load(model_weights_path, map_location='cpu')) # Load to CPU first
    float_model.to(DEVICE)
    float_model.eval() # IMPORTANT: Set to evaluation mode
    print("Float model loaded successfully.")
    return float_model

# Dataset class adapted for Parquet/Pandas tabular data
class CalibrationDataset(Dataset):
    def __init__(self, x_parquet_path, subset_len=None):
        print(f"Loading calibration data from: {x_parquet_path}")
        if not os.path.exists(x_parquet_path):
             raise FileNotFoundError(f"Calibration data file not found: {x_parquet_path}")

        # Your data preprocessing steps
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

        X_train = pd.read_parquet(x_parquet_path)
        # Careful: Ensure these operations are safe and intended for *all* uses of this dataset class
        # Dropping index[0] might be specific to how the parquet file was created.
        if not X_train.empty:
            X_train = X_train.astype(datatypes)
            X_train.columns = col_names
            X_train = X_train.drop(X_train.index[0])
        else:
            print("Warning: Loaded DataFrame is empty.")
            self.X = pd.DataFrame() # Use empty DataFrame
            return # Skip further processing

        if subset_len:
            if subset_len > len(X_train):
                print(f"Warning: subset_len ({subset_len}) > dataset size ({len(X_train)}). Using full dataset.")
                self.X = X_train
            else:
                # Select the first subset_len samples for calibration
                self.X = X_train.iloc[:subset_len]
        else:
            # Use the full (preprocessed) dataset if subset_len is None
            self.X = X_train

        print(f"Loaded {len(self.X)} samples for calibration.")
        # Labels are not strictly needed for PTQ calibration, so we won't load y_train here
        # If you needed them for some reason, load y_parquet_path similarly

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Get data sample as numpy array
        x_sample = self.X.iloc[idx].values.astype(np.float32)
        # Convert to PyTorch FloatTensor
        x_tensor = torch.from_numpy(x_sample)

        # Return data and a dummy label (0) as DataLoader expects tuples
        return x_tensor, torch.tensor(0)

def get_calibration_loader(x_parquet_path, batch_size, subset_len):
    """Creates DataLoader for calibration using Parquet/Pandas data."""
    dataset = CalibrationDataset(x_parquet_path=x_parquet_path, subset_len=subset_len)
    if len(dataset) == 0:
        raise ValueError(f"No data loaded for calibration from: {x_parquet_path}")

    # Adjust num_workers based on your system (0 might be safer for Pandas integration)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Calibration DataLoader created with {len(dataset)} samples.")
    return loader

# --- Main Quantization Logic ---
def main():
    parser = argparse.ArgumentParser(description='Vitis AI PTQ Calibration Script for Tabular Data')
    # Model Args
    parser.add_argument('--model_def', type=str, required=True, help='Path to Python file defining the model class (e.g., my_model.py).')
    parser.add_argument('--model_class', type=str, required=True, help='Name of the model class (e.g., DeepNN).')
    parser.add_argument('--model_weights', type=str, required=True, help='Path to the float model .pth weights file.')
    # Model Hyperparameters
    parser.add_argument('--input_size', type=int, default=43, help='Model input size (number of features).')
    parser.add_argument('--hidden_size', type=int, required=True, help='Number of neurons in hidden layers.')
    parser.add_argument('--num_hidden_layers', type=int, required=True, help='Total number of Linear layers in the model.')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of output classes.')
    # Data Args
    parser.add_argument('--x_calib_data', type=str, required=True, help='Path to calibration data features (X_train.parquet).')
    # Quantization Args
    parser.add_argument('--output_dir', type=str, default='./quantize_result', help='Directory to save quantization results.')
    parser.add_argument('--subset_len', type=int, default=100, help='Number of samples for calibration.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for calibration.')
    parser.add_argument('--deploy', action='store_true', help='Export .xmodel and related files after calibration (Mandatory for deployment).')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Model ---
    float_model = load_model_from_file(
        args.model_def, args.model_weights, args.model_class,
        args.input_size, args.hidden_size, args.num_hidden_layers, args.num_classes
    )

    # --- Get Calibration Dataloader ---
    calib_loader = get_calibration_loader(args.x_calib_data, args.batch_size, args.subset_len)

    # --- Run Calibration ---
    print("\n--- Running Quantization Calibration ---")

    # Prepare a dummy input tensor for the quantizer
    # Shape is (batch_size, num_features) - use batch_size=1 for simplicity here
    # The actual batch size during calibration comes from the DataLoader
    dummy_input_tensor = torch.randn([1] + list(INPUT_SHAPE)).to(DEVICE)

    # Instantiate the Vitis AI quantizer for calibration mode
    quantizer = torch_quantizer(
        quant_mode='calib',           # Fixed to calibration
        module=float_model,           # Your float model instance
        input_args=(dummy_input_tensor,), # Sample input tensor (shape matters)
        output_dir=args.output_dir,
        device=DEVICE
        # bitwidth=8,                 # Default: 8
        # method='diffs'              # Default: 'diffs'
    )

    # Get the quantized model wrapper (needed for calibration pass)
    quantized_model = quantizer.quant_model
    quantized_model.eval() # Ensure evaluation mode

    print(f"\n--- Starting Calibration Pass ({args.subset_len} samples target) ---")
    # Run calibration forward passes using the calibration loader
    with torch.no_grad():
         # Use only subset_len samples as loaded by the Dataset class
        for i, (cal_data, _) in enumerate(calib_loader):
            cal_data = cal_data.to(DEVICE)
            _ = quantized_model(cal_data) # Run forward pass
            print(f"\rCalibration Progress: Batch {i+1}/{len(calib_loader)}", end='')
            # Stop early if subset_len is smaller than total dataset size *in the loader*
            # This logic might need adjustment if subset_len is handled solely in Dataset
            if (i + 1) * args.batch_size >= args.subset_len:
                 print(f"\nReached target subset_len ({args.subset_len}). Stopping calibration pass.")
                 break
    print("\nCalibration Forward Pass Complete.")

    # Export quantization configuration (Quant_info.json) - CRITICAL
    print("Exporting quantization configuration...")
    quantizer.export_quant_config()
    print(f"Quantization config saved in {args.output_dir}")

    # Export for deployment if requested
    if args.deploy:
        print("Exporting model for deployment (.xmodel)...")
        # export_xmodel is the typical function for DPU targets
        try:
             quantizer.export_xmodel(output_dir=args.output_dir, deploy_check=True)
             print(f".xmodel and related files exported to {args.output_dir}")
        except Exception as e:
             print(f"ERROR during export_xmodel: {e}")
             print("Deployment artifact export failed.")
    else:
        print("Skipping deployment export (--deploy flag not set).")
        print("Run again with --deploy to generate .xmodel for compilation.")

    print("\nQuantization script finished.")


if __name__ == '__main__':
    main()
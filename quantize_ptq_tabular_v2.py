# quantize_ptq_tabular_v2.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import argparse
import os
import importlib.util # To load model definition dynamically
from tqdm import tqdm # For progress bar during calibration

# Import the Vitis AI Quantizer API
from pytorch_nndct.apis import torch_quantizer

# --- Configuration ---
# Input shape for a single sample (features,)
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
        if not X_train.empty:
            X_train = X_train.astype(datatypes)
            X_train.columns = col_names
            if 0 in X_train.index: # Check if index 0 exists before dropping
               X_train = X_train.drop(X_train.index[0])
        else:
            print("Warning: Loaded DataFrame is empty.")
            self.X = pd.DataFrame()
            return

        if subset_len:
            if subset_len > len(X_train):
                print(f"Warning: subset_len ({subset_len}) > dataset size ({len(X_train)}). Using full dataset for calibration.")
                self.X = X_train
            else:
                self.X = X_train.iloc[:subset_len]
        else:
            # Use the full (preprocessed) dataset if subset_len is None (for calibration)
            print("Warning: subset_len not set, using full dataset for calibration.")
            self.X = X_train

        print(f"Loaded {len(self.X)} samples for calibration.")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X.iloc[idx].values.astype(np.float32)
        x_tensor = torch.from_numpy(x_sample)
        # Return data and a dummy label (0)
        return x_tensor, torch.tensor(0)

def get_calibration_loader(x_parquet_path, batch_size, subset_len):
    """Creates DataLoader for calibration using Parquet/Pandas data."""
    dataset = CalibrationDataset(x_parquet_path=x_parquet_path, subset_len=subset_len)
    if len(dataset) == 0:
        raise ValueError(f"No data loaded for calibration from: {x_parquet_path}")

    # num_workers=0 can be safer for Pandas/multiprocessing interactions
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"Calibration DataLoader created with {len(dataset)} samples.")
    return loader

# --- Main Quantization Logic ---
def main():
    parser = argparse.ArgumentParser(description='Vitis AI PTQ Calibration & Deployment Script for Tabular Data')
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
    parser.add_argument('--output_dir', type=str, default='./quantize_result', help='Directory for quantization results (quant_info.json, exports).')
    parser.add_argument('--subset_len', type=int, default=1000, help='Number of samples for calibration (used in calib mode).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for calibration (used in calib mode).')
    parser.add_argument('--quant_mode', type=str, default='calib', choices=['calib', 'test'], help="'calib': run quantization calibration, 'test': load calibration result and optionally deploy.")
    parser.add_argument('--deploy', action='store_true', help="Export deployment artifacts (.xmodel, etc.) in 'test' mode.")

    args = parser.parse_args()

    # --- Check Mode Compatibility ---
    if args.quant_mode != 'test' and args.deploy:
        print("Warning: Deployment (--deploy) requires 'test' mode. Disabling --deploy for this 'calib' run.")
        args.deploy = False # Turn off deploy if not in test mode

    # --- Create output directory ---
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load Float Model ---
    # Loaded in both modes because quantizer needs the original model definition
    float_model = load_model_from_file(
        args.model_def, args.model_weights, args.model_class,
        args.input_size, args.hidden_size, args.num_hidden_layers, args.num_classes
    )

    # --- Instantiate Quantizer ---
    # Prepare a dummy input tensor for the quantizer shape inference (batch_size=1 is fine)
    dummy_input_tensor = torch.randn([1] + list(INPUT_SHAPE)).to(DEVICE)

    quantizer = torch_quantizer(
        quant_mode=args.quant_mode,   # 'calib' or 'test'
        module=float_model,           # Pass the float model structure
        input_args=(dummy_input_tensor,), # Provide sample input shape
        output_dir=args.output_dir,   # Directory for intermediate files & quant_info.json
        device=DEVICE
    )

    # Get the quantized model wrapper
    quantized_model = quantizer.quant_model

    # --- Run Selected Mode ---

    if args.quant_mode == 'calib':
        print(f"\n--- Running Quantization Calibration ---")
        print(f"Using {args.subset_len} samples with batch size {args.batch_size}")

        # Get calibration dataloader
        calib_loader = get_calibration_loader(args.x_calib_data, args.batch_size, args.subset_len)

        # Run calibration forward passes
        quantized_model.eval()
        with torch.no_grad():
            for i, (cal_data, _) in tqdm(enumerate(calib_loader), total=len(calib_loader), desc="Calibration"):
                cal_data = cal_data.to(DEVICE)
                _ = quantized_model(cal_data) # Run forward pass to gather stats

        print("\nCalibration Forward Pass Complete.")

        # Export quantization configuration (Quant_info.json) - CRITICAL
        print("Exporting quantization configuration...")
        quantizer.export_quant_config()
        print(f"Quantization config saved: {os.path.join(args.output_dir, 'quant_info.json')}")
        print("\nCalibration finished. Run again with quant_mode='test' --deploy to generate deployment models.")

    elif args.quant_mode == 'test':
        print("\n--- Running Quantization Test Mode ---")

        # Check if calibration results exist (needed for test mode)
        quant_info_path = os.path.join(args.output_dir, 'quant_info.json')
        if not os.path.exists(quant_info_path):
             print(f"Error: Calibration results '{quant_info_path}' not found.")
             print(f"Please run with quant_mode='calib' first.")
             return

        # <<< START INSERTED CODE >>>
        # Perform a dummy forward pass to finalize the graph for export
        print("Performing a dummy forward pass in 'test' mode before export...")
        quantized_model.eval() # Ensure eval mode
        with torch.no_grad():
            # Create a dummy input with batch size 1
            dummy_test_input = torch.randn([1] + list(INPUT_SHAPE)).to(DEVICE)
            try:
                _ = quantized_model(dummy_test_input) # The crucial forward pass
                print("Dummy forward pass successful.")
            except Exception as e:
                print(f"ERROR during dummy forward pass: {e}")
                print("Cannot proceed to export.")
                return # Exit if the forward pass itself fails
        # <<< END INSERTED CODE >>>

        if args.deploy:
            print("Attempting deployment exports...")

            export_ok = True
            # --- Export TorchScript ---
            try:
                print("Exporting TorchScript model...")
                quantizer.export_torch_script(output_dir=args.output_dir)
                print(" -> TorchScript export seems successful (check logs/output dir).")
            except Exception as e:
                print(f" -> ERROR during TorchScript export: {e}")
                export_ok = False

            # --- Export ONNX ---
            try:
                print("Exporting ONNX model...")
                quantizer.export_onnx_model(output_dir=args.output_dir)
                print(" -> ONNX export seems successful (check logs/output dir).")
            except Exception as e:
                print(f" -> ERROR during ONNX export: {e}")
                export_ok = False

            # --- Export XModel ---
            try:
                print("Exporting XModel...")
                # Using deploy_check=False based on previous findings
                quantizer.export_xmodel(output_dir=args.output_dir, deploy_check=False)
                print(" -> XModel export seems successful (check logs/output dir).")
                # Explicitly check for the xmodel file after the call
                model_base_name = args.model_class
                xmodel_path_expected = os.path.join(args.output_dir, f"{model_base_name}_int.xmodel") # Adjust name if needed
                if os.path.exists(xmodel_path_expected):
                    print(f"   -> Verified: .xmodel file found at {xmodel_path_expected}")
                else:
                    print(f"   -> WARNING: .xmodel file NOT found at {xmodel_path_expected} despite no Python error!")
                    export_ok = False # Mark as failed if file not present
            except Exception as e:
                print(f" -> ERROR during XModel export: {e}")
                import traceback
                traceback.print_exc()
                export_ok = False

            # --- Summary ---
            if not export_ok:
                print("\nWarning: One or more deployment exports may have failed or produced errors. Please check logs and output directory carefully.")
            else:
                 print("\nDeployment exports completed. Please check output directory for artifacts.")

        else:
            print("Running in 'test' mode without --deploy flag.")
            print("Quantized model is loaded, dummy forward pass executed, but no deployment artifacts generated.")

    print("\nQuantization script finished.")


if __name__ == '__main__':
    main()

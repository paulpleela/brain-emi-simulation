"""
Load FD tensor data from individual scenario NPZ files and aggregate into train/val/test splits.
"""
import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    metadata = pd.read_csv("dataset_metadata.csv")
    
    # Pre-allocate lists per split
    data = {
        "train": {"X": [], "y": []},
        "val": {"X": [], "y": []},
        "test": {"X": [], "y": []},
    }
    
    # Make sure output directory exists
    os.makedirs("deep_learning/datasets", exist_ok=True)
    
    files = sorted(glob.glob("fd_tensors/scenario_*_fd.npz"))
    print(f"Found {len(files)} FD tensor files")
    
    for f in tqdm(files):
        basename = os.path.basename(f)
        try:
            scenario_id = int(basename.split('_')[1])
        except (ValueError, IndexError):
            continue
        
        row = metadata[metadata["scenario_id"] == scenario_id]
        if len(row) == 0:
            continue
        
        split = row.iloc[0]["split"]
        label = row.iloc[0]["has_lesion"]
        
        # Load FD tensor
        tensor_data = np.load(f)
        signal = tensor_data['signal']  # shape (512, 90)
        
        if split in data:
            data[split]["X"].append(signal)
            data[split]["y"].append(label)
    
    # Convert to numpy arrays
    print("Converting to numpy arrays...")
    X_train = np.array(data["train"]["X"])
    y_train = np.array(data["train"]["y"])
    X_val = np.array(data["val"]["X"])
    y_val = np.array(data["val"]["y"])
    X_test = np.array(data["test"]["X"]) if len(data["test"]["X"]) > 0 else np.array([])
    y_test = np.array(data["test"]["y"]) if len(data["test"]["y"]) > 0 else np.array([])
    
    # Save
    save_dict = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
    }
    if len(X_test) > 0:
        save_dict["X_test"] = X_test
        save_dict["y_test"] = y_test
    
    out_path = "deep_learning/datasets/dataset_fd_tensors.npz"
    np.savez_compressed(out_path, **save_dict)
    print(f"Saved {out_path}")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    if len(X_test) > 0:
        print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

if __name__ == "__main__":
    main()

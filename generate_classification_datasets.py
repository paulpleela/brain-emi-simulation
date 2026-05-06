import os
import glob
import pandas as pd
import numpy as np
import skrf as rf
from tqdm import tqdm

def process_file(filepath):
    # Returns ntwk.s shape (F, 16, 16) complex
    try:
        ntwk = rf.Network(filepath)
        return ntwk.s
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    metadata = pd.read_csv("dataset_metadata.csv")
    
    # Pre-allocate lists
    data = {
        "real_imag": {"train": [], "val": [], "test": []},
        "mag_phase": {"train": [], "val": [], "test": []},
        "mag_only": {"train": [], "val": [], "test": []}
    }
    labels = {"train": [], "val": [], "test": []}
    
    # Make sure output directory exists
    os.makedirs("deep_learning/datasets", exist_ok=True)
    
    files = glob.glob("sparams/scenario_*.s16p")
    for f in tqdm(files):
        # Extract scenario ID
        basename = os.path.basename(f)
        try:
            scenario_id = int(basename.split('_')[1].split('.')[0])
        except ValueError:
            continue
            
        row = metadata[metadata["scenario_id"] == scenario_id]
        if len(row) == 0:
            continue
        
        split = row.iloc[0]["split"]
        label = row.iloc[0]["has_lesion"]
        
        S = process_file(f)
        if S is None:
            continue
            
        # S is of shape (F, 16, 16)
        
        # 1. Real / Imaginary
        real_imag = np.stack([np.real(S), np.imag(S)], axis=-1)
        
        # 2. Magnitude + sin/cos(phase)
        mag = np.abs(S)
        phase = np.angle(S)
        mag_phase = np.stack([mag, np.cos(phase), np.sin(phase)], axis=-1)
        
        # 3. Magnitude only
        mag_only = np.expand_dims(mag, axis=-1)
        
        if split not in ["train", "val", "test"]:
            split = "train"  # Fallback
            
        data["real_imag"][split].append(real_imag)
        data["mag_phase"][split].append(mag_phase)
        data["mag_only"][split].append(mag_only)
        labels[split].append(label)
        
    print("Saving datasets...")
    for format_key, format_data in data.items():
        save_dict = {}
        for split in ["train", "val", "test"]:
            if len(format_data[split]) > 0:
                save_dict[f"X_{split}"] = np.array(format_data[split])
                save_dict[f"y_{split}"] = np.array(labels[split])
        
        out_path = f"deep_learning/datasets/dataset_{format_key}.npz"
        np.savez_compressed(out_path, **save_dict)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
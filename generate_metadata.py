"""
Generate metadata CSV for machine learning dataset

Creates a CSV file with scenario information for ML training:
- Scenario ID
- Lesion presence (0=healthy, 1=hemorrhage)
- Lesion size (mm)
- Lesion position (x, y, z in meters)
- Dataset split (train/val/test)
"""

import os
import re
import csv

INPUT_DIR = "brain_inputs"
OUTPUT_FILE = "dataset_metadata.csv"

print("Parsing scenario files...")
scenarios = {}

# Parse all input files to extract metadata
for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.startswith("scenario_") or not filename.endswith(".in"):
        continue
    
    # Extract scenario ID and transmit antenna
    match = re.match(r'scenario_(\d+)_tx(\d+)\.in', filename)
    if not match:
        continue
    
    scenario_id = int(match.group(1))
    tx_id = int(match.group(2))
    
    # Only process tx01 files (metadata same for all tx in scenario)
    if tx_id != 1:
        continue
    
    # Read file to extract hemorrhage info
    filepath = os.path.join(INPUT_DIR, filename)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header comments
    has_lesion = 0
    lesion_size_mm = 0
    lesion_x, lesion_y, lesion_z = 0, 0, 0
    
    for line in lines[:10]:
        if "Healthy baseline" in line:
            has_lesion = 0
            break
        elif "Hemorrhage:" in line:
            has_lesion = 1
            # Parse "Hemorrhage: 5mm at (x, y, z)"
            size_match = re.search(r'(\d+)mm', line)
            if size_match:
                lesion_size_mm = int(size_match.group(1))
            
            # Parse position - handle numpy float formatting
            pos_match = re.search(r'at \((.*?)\)', line)
            if pos_match:
                pos_str = pos_match.group(1)
                # Remove numpy formatting
                pos_str = pos_str.replace('np.float64(', '').replace(')', '')
                coords = [float(x.strip()) for x in pos_str.split(',')]
                if len(coords) == 3:
                    lesion_x, lesion_y, lesion_z = coords
            break
    
    scenarios[scenario_id] = {
        'scenario_id': scenario_id,
        'has_lesion': has_lesion,
        'lesion_size_mm': lesion_size_mm,
        'lesion_x': lesion_x,
        'lesion_y': lesion_y,
        'lesion_z': lesion_z
    }

print(f"Found {len(scenarios)} scenarios")

# Assign dataset splits (70% train, 15% val, 15% test)
scenario_ids = sorted(scenarios.keys())
n_total = len(scenario_ids)
n_train = int(0.70 * n_total)
n_val = int(0.15 * n_total)
n_test = n_total - n_train - n_val

# Stratify by lesion presence
healthy_ids = [sid for sid in scenario_ids if scenarios[sid]['has_lesion'] == 0]
lesion_ids = [sid for sid in scenario_ids if scenarios[sid]['has_lesion'] == 1]

# Split each group
def split_list(lst, ratios):
    n = len(lst)
    n1 = int(ratios[0] * n)
    n2 = int(ratios[1] * n)
    return lst[:n1], lst[n1:n1+n2], lst[n1+n2:]

healthy_train, healthy_val, healthy_test = split_list(healthy_ids, (0.70, 0.15))
lesion_train, lesion_val, lesion_test = split_list(lesion_ids, (0.70, 0.15))

# Assign splits
for sid in healthy_train + lesion_train:
    scenarios[sid]['split'] = 'train'
for sid in healthy_val + lesion_val:
    scenarios[sid]['split'] = 'val'
for sid in healthy_test + lesion_test:
    scenarios[sid]['split'] = 'test'

# Write CSV
print(f"Writing {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', newline='') as f:
    fieldnames = ['scenario_id', 'has_lesion', 'lesion_size_mm', 
                  'lesion_x', 'lesion_y', 'lesion_z', 'split']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    
    for sid in sorted(scenarios.keys()):
        writer.writerow(scenarios[sid])

print("Complete!")
print(f"\nDataset summary:")
print(f"  Healthy: {len(healthy_ids)} scenarios")
print(f"    Train: {len(healthy_train)}, Val: {len(healthy_val)}, Test: {len(healthy_test)}")
print(f"  Hemorrhage: {len(lesion_ids)} scenarios")
print(f"    Train: {len(lesion_train)}, Val: {len(lesion_val)}, Test: {len(lesion_test)}")
print(f"  Total: {len(scenarios)} scenarios")
print(f"    Train: {len(healthy_train)+len(lesion_train)}")
print(f"    Val: {len(healthy_val)+len(lesion_val)}")
print(f"    Test: {len(healthy_test)+len(lesion_test)}")

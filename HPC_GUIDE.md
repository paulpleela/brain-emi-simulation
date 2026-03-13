# Running Brain EMI Simulations on Rangpur HPC


## Part 1: Set Up on HPC (First Time Only)

### Clone or Pull Repository:

```bash
# If first time - clone repository
cd ~
git clone https://github.com/paulpleela/brain-emi-simulation.git
cd brain-emi-simulation

# If already exists - pull latest changes
cd ~/brain-emi-simulation
git pull origin main
```

### Clone and Install gprMax:

```bash
# Clone official gprMax (not included in our repo)
cd ~/brain-emi-simulation
git clone https://github.com/gprMax/gprMax.git

# Navigate to gprMax directory
cd gprMax

# Create conda environment
conda env create -f conda_env.yml

# Activate environment
conda activate gprmax

# Build and install gprMax (takes 5-10 minutes)
python setup.py build
python setup.py install

# Verify installation
python -c "import gprMax; print('gprMax installed successfully!')"

# Return to project root
cd ~/brain-emi-simulation
```

**Expected output**: `gprMax installed successfully!`

---

## Part 2: Test Single Scenario

### Prepare Test:

```bash
cd ~/brain-emi-simulation

# Create logs directory
mkdir -p logs

# Check you have input files
ls brain_inputs/ | head -5
# Should see: scenario_001_tx01.in, scenario_001_tx02.in, etc.
```

### Run Single Test Job:

```bash
# Test one file manually
conda activate gprmax
python -m gprMax brain_inputs/scenario_001_tx01.in -n 8
```

**Expected:**
- Runtime: 45-60 minutes
- Output: `brain_inputs/scenario_001_tx01.out` (~50-100 MB)

If this works, your setup is correct! ✅

---

## Part 2b: Enable GPU Acceleration (Recommended)

gprMax supports CUDA GPU execution via the `-gpu` flag. On Rangpur this gives **~8–12× speedup** per job (3–5 min vs 45–60 min on CPU).

### Install pycuda in the gprmax environment:

```bash
conda activate gprmax
pip install pycuda
```

**Expected output ends with**: `Successfully installed pycuda-...`

> **Note**: pycuda compiles CUDA kernels at first run. This takes ~30 seconds on first execution — normal behaviour.

### Check GPU availability on Rangpur:

```bash
# See GPU partitions and nodes
sinfo -p a100
sinfo -p a100-test

# Quick interactive test session (a100-test is low wait time, 20 min limit)
srun -p a100-test --gres=shard:1 --pty bash
nvidia-smi
exit
```

> **Rangpur GPU partitions:**
> - `a100-test` — for development/testing, short wait, 20 min time limit, use `--gres=shard:1`
> - `a100` — for production runs, full A100 40GB GPU, use `--gres=gpu:1`
> - `p100` — older P100 GPUs (only needed for old CUDA versions), use `--gres=gpu:1`

### Test single GPU job:

```bash
# Get an interactive session on a100-test (low wait, 20 min limit)
srun -p a100-test --gres=shard:1 --pty bash

# Inside the session:
conda activate gprmax
cd ~/brain-emi-simulation
python -m gprMax brain_inputs/scenario_002_tx01.in -n 1 -gpu
exit
```

**Expected:**
- Runtime: **3–5 minutes** (vs 45–60 min on CPU)
- First run prints CUDA kernel compilation messages — this is normal
- Output: `brain_inputs/scenario_001_tx01.out`

If this works, you're ready to submit the GPU job array. ✅

### CPU vs GPU comparison:

| | CPU (8 cores) | GPU (CUDA) |
|---|---|---|
| Time per job | 45–60 min | 3–5 min |
| Speedup | 1× | ~8–12× |
| SLURM partition | `cpu` | `a100` |
| SLURM resource | `--cpus-per-task=8` | `--gres=gpu:1` |
| gprMax flag | *(none)* | `-gpu` |
| Script to use | `run_simulation.sh` | `run_simulation_gpu.sh` |
| 4800 jobs at 16 parallel | ~19 hours | ~2.5 hours |
| 4800 jobs at 32 parallel | ~11 hours | **~1.5 hours** |

---

## Part 3: Run Small Batch (10 Scenarios)

### Modify SLURM Script:

```bash
cd ~/brain-emi-simulation

# Edit run_simulation.sh
nano run_simulation.sh

# Change line 5 from:
#SBATCH --array=1-4800%16

# To (for 10 scenarios = 160 jobs):
#SBATCH --array=1-160%16

# Save: Ctrl+O, Enter, Ctrl+X
```

### Submit Batch Job:

```bash
# Make sure you're in the right directory
cd ~/brain-emi-simulation

# Submit GPU job array (recommended)
sbatch run_simulation_gpu.sh

# Or CPU job array (fallback if no GPU nodes available)
# sbatch run_simulation.sh

# You'll see: "Submitted batch job 123456"
```

### Monitor Jobs:

```bash
# Check your jobs
squeue -u s4910027

# Watch in real-time (updates every 10 seconds)
watch -n 10 'squeue -u s4910027'
# Press Ctrl+C to exit

# Check completed jobs
sacct --format=JobID,JobName,State,Elapsed,MaxRSS

# Count completed
sacct --format=JobID,State | grep COMPLETED | wc -l
```

**Expected:**
- 16 jobs running at once (parallel limit)
- Total runtime: ~1 hour for 160 jobs
- Output: 160 `.out` files in `brain_inputs/`

---

## Part 4: Run Full Dataset (300 Scenarios)

### Update SLURM Script:

```bash
cd ~/brain-emi-simulation

# Edit run_simulation.sh
nano run_simulation.sh

# Change line 5 back to:
#SBATCH --array=1-4800%16

# Save and exit
```

### Submit Full Batch:

```bash
# Submit all 300 scenarios (4800 jobs) using GPU (recommended)
sbatch run_simulation_gpu.sh

# Monitor progress
watch -n 60 'echo "Running: $(squeue -u s4910027 | wc -l)"; echo "Completed: $(sacct --format=State | grep COMPLETED | wc -l)"'
```

**Expected (GPU):**
- 4800 jobs total
- 32 running in parallel at any time
- Total runtime: **~1.5 hours** (vs ~19 hours on CPU)
- Output: 4800 `.out` files (~380 GB total)

---

## Part 5: Check Results

### Verify Outputs:

```bash
cd ~/brain-emi-simulation/brain_inputs

# Count output files
ls *.out | wc -l
# Should be: 4800

# Check file sizes
ls -lh scenario_001_tx01.out
# Should be: 50-100 MB

# Check a few outputs
ls -lh scenario_*.out | head -20
```

### Check for Failures:

```bash
cd ~/brain-emi-simulation

# Check for failed jobs
sacct --format=JobID,State | grep FAILED

# Check error logs
ls logs/*.err | xargs grep -l ERROR
```

---

## Troubleshooting

### Job Failed?

```bash
# Find failed job ID
sacct --format=JobID,State | grep FAILED

# Check error log (replace 123456 with job ID)
cat logs/sim_123456.err
```

### Common Issues:

**1. Conda environment not found:**
```bash
conda env list  # Check it exists
conda activate gprmax  # Activate it
```

**2. gprMax not installed:**
```bash
conda activate gprmax
cd ~/brain-emi-simulation/gprMax
python setup.py install
```

**3. Out of memory:**
```bash
# Edit run_simulation.sh, add after line 8:
#SBATCH --mem=16G
```

**4. Timeout (job killed after 1 hour):**
```bash
# Edit run_simulation.sh, change line 7:
#SBATCH --time=02:00:00
```

**5. Input file not found:**
```bash
# Make sure you're in right directory
cd ~/brain-emi-simulation
ls brain_inputs/ | head
```

**6. GPU not found / pycuda error:**
```bash
# Verify pycuda is installed
conda activate gprmax
python -c "import pycuda; print('pycuda OK')"

# If not installed:
pip install pycuda

# Verify CUDA is available
nvidia-smi

# Test GPU manually
python -m gprMax brain_inputs/scenario_001_tx01.in -n 1 -gpu

# If GPU nodes are busy/unavailable, fall back to CPU:
sbatch run_simulation.sh   # (uses --partition=cpu)
```

### On Rangpur HPC:
```bash
# Update code
cd ~/brain-emi-simulation
git pull origin main

# Activate environment
conda activate gprmax

# Submit job
sbatch run_simulation.sh

# Monitor
squeue -u s4910027
watch -n 10 'squeue -u s4910027'

# Check results
sacct --format=JobID,State,Elapsed
ls brain_inputs/*.out | wc -l
```

---

## Timeline Estimate

**CPU only:**
- **Setup** (first time): 30 minutes
- **Single test**: 45–60 minutes
- **Full dataset** (300 scenarios, 16 parallel): ~19 hours

**With GPU (recommended):**
- **Setup** (first time): 30 minutes + 5 min pycuda install
- **Single test**: 3–5 minutes
- **Full dataset** (300 scenarios, 32 parallel): **~1.5 hours**

**Total from scratch (GPU):** ~2 hours

---

## File Locations on HPC

```
~/brain-emi-simulation/
├── brain_inputs/              # 4800 input files + output files
│   ├── scenario_001_tx01.in
│   ├── scenario_001_tx01.out  # Created after simulation
│   └── ...
├── logs/                      # SLURM output logs
│   ├── sim_1.out
│   ├── sim_1.err
│   └── ...
├── gprMax/                    # Source code
├── run_simulation.sh          # SLURM batch script
├── generate_dataset.py        # Dataset generator
└── dataset_metadata.csv       # Scenario info for ML
```

---

## Next Steps After Simulation

1. **Extract S-parameters** from `.out` files
2. **Create S-matrix dataset** (16×16 for each scenario)
3. **Train ML model** using `dataset_metadata.csv`
4. **Clean up** large `.out` files after extraction

---

**Ready to start!** Follow steps 1-4 to get set up, then run your simulations. 🚀

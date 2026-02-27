# Running Brain EMI Simulations on Rangpur HPC

Complete guide from git push to running 300 scenarios.

---

## Part 1: Push Your Code to GitHub

### On your local machine (Windows PowerShell):

```powershell
# Navigate to your project
cd "C:\Users\paudo\OneDrive\Documents\Thesis gprmax"

# Check what's new
git status

# Add all new files
git add .

# Commit with message
git commit -m "Add 300-scenario dataset for ML training"

# Push to GitHub
git push origin main
```

---

## Part 2: Connect to Rangpur HPC

### SSH to Rangpur:

```powershell
ssh s4910027@login0.hpc.griffith.edu.au
```

Enter your password when prompted.

---

## Part 3: Set Up on HPC (First Time Only)

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

### Create Conda Environment:

```bash
# Load conda module (if needed on Rangpur)
module load anaconda3  # or conda, check with: module avail

# Create gprMax environment
cd gprMax
conda env create -f conda_env.yml

# This creates an environment named "gprMax" with all dependencies
```

### Build gprMax:

```bash
# Activate the environment
conda activate gprmax

# Build gprMax
cd ~/brain-emi-simulation/gprMax
python setup.py build
python setup.py install

# Verify installation
python -c "import gprMax; print('gprMax installed successfully!')"
```

---

## Part 4: Test Single Scenario

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

## Part 5: Run Small Batch (10 Scenarios)

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

# Submit job array
sbatch run_simulation.sh

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

## Part 6: Run Full Dataset (300 Scenarios)

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
# Submit all 300 scenarios (4800 jobs)
sbatch run_simulation.sh

# Monitor progress
watch -n 60 'echo "Running: $(squeue -u s4910027 | wc -l)"; echo "Completed: $(sacct --format=State | grep COMPLETED | wc -l)"'
```

**Expected:**
- 4800 jobs total
- 16 running in parallel at any time
- Total runtime: ~19 hours (300 hours / 16 parallel)
- Output: 4800 `.out` files (~380 GB total)

---

## Part 7: Check Results

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

## Part 8: Download Results (Optional)

### Download to Local Machine:

From your Windows PowerShell:

```powershell
# Create local output directory
mkdir outputs

# Download all .out files (WARNING: ~380 GB!)
scp -r s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/brain_inputs/*.out ./outputs/

# Or download specific scenarios
scp s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/brain_inputs/scenario_001_*.out ./outputs/
```

**Tip:** Process S-parameters on HPC first to reduce download size!

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

---

## Quick Reference Commands

### On Local Machine:
```powershell
# Push code
git add .
git commit -m "Update"
git push origin main

# SSH to HPC
ssh s4910027@login0.hpc.griffith.edu.au
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

- **Setup** (first time): 30 minutes
- **Single test**: 45-60 minutes
- **Small batch** (10 scenarios): 1 hour
- **Full dataset** (300 scenarios): 19 hours

**Total from scratch:** ~21 hours

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

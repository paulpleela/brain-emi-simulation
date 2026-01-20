# HPC Test Run Instructions - Rangpur

## Quick Start (Single Scenario Test)

### Step 1: Upload Files to HPC

Open PowerShell and run:

```powershell
# Upload test files
scp -r test_single test_single_job.sh s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# Upload gprMax source (if not already there)
# scp -r gprMax s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/
```

### Step 2: SSH to Rangpur

```powershell
ssh s4910027@login0.hpc.griffith.edu.au
```

### Step 3: Navigate and Submit Job

```bash
cd ~/brain-emi-simulation

# Create logs directory
mkdir -p logs

# Check conda environment exists
conda env list

# If gprmax environment doesn't exist, create it:
# conda env create -f gprMax/conda_env.yml

# Submit test job
sbatch test_single_job.sh
```

### Step 4: Monitor Job

```bash
# Check job status
squeue -u s4910027

# Watch job queue (updates every 10 seconds)
watch -n 10 'squeue -u s4910027'

# Check job history
sacct --format=JobID,JobName,State,Elapsed,MaxRSS

# View output (while running or after completion)
tail -f test_single/test.err
tail -f test_single/test.out
```

### Step 5: Check Results

```bash
# After job completes, check output file
ls -lh test_single/*.out

# Should see: scenario_001_tx01.out (~50-100 MB)

# View simulation log
cat test_single/test.out

# Check for errors
cat test_single/test.err
```

### Step 6: Download Results

Exit SSH and run locally:

```powershell
# Download output file
scp s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/test_single/scenario_001_tx01.out ./

# Download logs
scp s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/test_single/test.out ./
scp s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/test_single/test.err ./
```

---

## Expected Results

### Success Indicators:

1. **Job completes** (state: COMPLETED in `sacct`)
2. **Output file created**: `scenario_001_tx01.out` exists
3. **File size**: ~50-100 MB (HDF5 format)
4. **No errors** in `test.err`
5. **Log shows**: "✓ SUCCESS - Ellipsoidal geometry works!"

### Typical Runtime:

- **Expected**: 45-60 minutes
- **CPUs**: 8 cores
- **Memory**: ~4-8 GB

---

## If Test Succeeds → Run Small Batch (10 Scenarios)

### Upload More Files:

```powershell
# Upload 10 scenarios (160 files: 10 scenarios × 16 tx)
scp -r brain_inputs s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# Update and upload batch script
scp run_simulation.sh s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/
```

### Modify run_simulation.sh:

```bash
# SSH to HPC
ssh s4910027@login0.hpc.griffith.edu.au
cd ~/brain-emi-simulation

# Edit line 5 to run first 10 scenarios (160 jobs)
nano run_simulation.sh
# Change: #SBATCH --array=1-4800%16
# To:     #SBATCH --array=1-160%16

# Submit
sbatch run_simulation.sh
```

### Monitor Batch:

```bash
# Check running jobs
squeue -u s4910027

# See completion rate
sacct --format=JobID,JobName,State,Elapsed | grep COMPLETED | wc -l

# Expected: 160 jobs complete in ~1 hour (16 parallel)
```

---

## If Small Batch Succeeds → Run Full Dataset (300 Scenarios)

### Full Run:

```bash
# Edit run_simulation.sh back to full array
nano run_simulation.sh
# Change: #SBATCH --array=1-160%16
# To:     #SBATCH --array=1-4800%16

# Submit all 300 scenarios
sbatch run_simulation.sh

# Monitor over ~19 hours
watch -n 60 'squeue -u s4910027; sacct --format=JobID,State | grep COMPLETED | wc -l'
```

---

## Troubleshooting

### Job Failed?

```bash
# Check error log
cat test_single/test.err

# Check SLURM output
cat test_single/test.out

# Check job details
scontrol show job JOBID
```

### Common Issues:

1. **Conda environment not found**
   - Solution: `conda env create -f gprMax/conda_env.yml`

2. **gprMax module not found**
   - Solution: `cd gprMax && python setup.py build && python setup.py install`

3. **Out of memory**
   - Solution: Add `#SBATCH --mem=16G` to job script

4. **Timeout (>1 hour)**
   - Solution: Increase `#SBATCH --time=02:00:00`

---

## File Locations on HPC

```
~/brain-emi-simulation/
├── test_single/
│   ├── scenario_001_tx01.in     # Input file
│   ├── scenario_001_tx01.out    # Output (after run)
│   ├── test.out                 # SLURM stdout
│   └── test.err                 # SLURM stderr
├── brain_inputs/                # Full dataset (4800 files)
├── gprMax/                      # Source code
├── test_single_job.sh           # Test job script
└── run_simulation.sh            # Full batch script
```

---

## Quick Commands Reference

```bash
# Upload
scp -r test_single test_single_job.sh s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/

# SSH
ssh s4910027@login0.hpc.griffith.edu.au

# Submit
sbatch test_single_job.sh

# Monitor
squeue -u s4910027
watch -n 10 'squeue -u s4910027'

# Check output
ls -lh test_single/*.out
cat test_single/test.out

# Download
scp s4910027@login0.hpc.griffith.edu.au:~/brain-emi-simulation/test_single/*.out ./

# Cancel job (if needed)
scancel JOBID
```

---

**Start with the single test!** If it works, you've validated the ellipsoidal geometry on HPC. Then scale up to 10 scenarios, then full 300. 🚀

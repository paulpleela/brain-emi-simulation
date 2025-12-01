# Realistic Brain Hemorrhage Imaging Model

## Overview

This document describes the improved, realistic simulation setup for brain hemorrhage detection using microwave imaging (0-2 GHz).

## Key Improvements from Original Model

### 1. Frequency Range: 0-2 GHz (vs. 0-130 GHz)
- **Monopole length**: 37.5 mm (λ/4 @ 2 GHz, down from 75 mm @ 1 GHz)
- **Ground plane**: 75×75 mm (smaller for 2 GHz operation)
- **Time window**: 15 ns (sufficient for head imaging at these frequencies)
- **Resolution**: Maintains 2 mm grid (still adequate: ~13 points/wavelength in brain @ 2 GHz)

### 2. Coupling Medium Layer
- **Purpose**: Improves electromagnetic coupling between antennas and head
- **Thickness**: 5 mm
- **Properties**: 
  - εᵣ = 32 (average of scalp and gray matter)
  - σ = 0.58 S/m
- **Benefit**: Reduces impedance mismatch, better signal penetration

### 3. Antenna Positioning
- **Configuration**: Fixed positions in equatorial ring around head
- **Distance**: Antennas touching coupling medium surface
- **Distribution**: 16 antennas equally spaced (22.5° apart)
- **Orientation**: Monopoles pointing upward (z-direction), ground planes horizontal

### 4. Head Model
- **Current implementation**: Concentric spheres with dimensions based on ellipsoid semi-axes
  - Average radius: ~9.5 cm (from a=9.5cm, b=7.5cm, c=11.5cm)
- **Layers**:
  - Coupling medium (outermost): ~110 mm radius
  - Scalp/skull: ~105 mm radius, 10 mm thick
  - Gray matter: ~95 mm radius, 3 mm thick  
  - White matter (core): ~92 mm radius
- **Note**: True ellipsoid geometry would require custom Python mesh generation (future enhancement)

### 5. Hemorrhagic Lesion
- **Location**: Offset from center (x-2cm, z+1cm) to simulate realistic hemorrhage position
- **Size**: Two overlapping blood spheres (15 mm + 10 mm diameter)
- **Material**: Blood tissue (εᵣ=61, σ=1.54 S/m)

## Tissue Electrical Properties (@ 1 GHz)

| Tissue | εᵣ | σ (S/m) | Source |
|--------|-----|---------|--------|
| Coupling medium | 32.0 | 0.58 | Average of scalp/gray matter |
| Scalp + skull | 12.0 | 0.20 | Literature values |
| Gray matter | 52.0 | 0.97 | Gabriel et al. |
| White matter | 38.0 | 0.57 | Gabriel et al. |
| Blood | 61.0 | 1.54 | Gabriel et al. |

## Simulation Parameters

### Domain
- **Size**: 600×600×600 mm (0.6 m cube)
- **Resolution**: 2 mm isotropic (dx = dy = dz = 2 mm)
- **Total cells**: 300×300×300 = 27 million cells
- **PML**: 10 cells (20 mm) on all boundaries

### Time Window
- **Duration**: 15 ns
- **Time step**: ~3.85 ps (CFL condition: dt = dx/(c√3))
- **Total steps**: ~3,896 time steps
- **Frequency resolution**: ~67 MHz (from 1/T)
- **Nyquist frequency**: ~130 GHz (adequate for 0-2 GHz analysis)

### Antenna Array
- **Type**: Quarter-wave monopoles with PEC ground planes
- **Quantity**: 16 antennas in equatorial ring
- **Port type**: 50Ω transmission line ports (#transmission_line)
- **Excitation**: Gaussian pulse (1 GHz center, 1 amplitude)
- **Receive termination**: 50Ω matched load (Gaussian waveform with 0 amplitude)

## File Structure

### Input Files
- **Directory**: `brain_monopole_realistic/`
- **Files**: `brain_realistic_tx01.in` through `brain_realistic_tx16.in`
- **Size**: ~350 lines per file, ~15 KB each
- **Content**: Full geometry + materials + all 16 antennas

### Output Files (after simulation)
- **Format**: HDF5 (.out)
- **Contents**: Voltage V(t) and current I(t) for each transmission line port
- **Size**: ~50-100 MB per file (depends on port data storage)
- **Location**: `/tls/` group in HDF5

### S-Parameter Files (after extraction)
- **Format**: Touchstone (.s16p)
- **Contents**: 16×16 S-matrix vs. frequency (0-2 GHz)
- **Columns**: Frequency | S11 | S12 | ... | S1616 (magnitude & phase)

## Computational Requirements

### Single Simulation
- **Cells**: 27 million
- **Time steps**: ~3,900
- **Memory**: ~8-12 GB RAM
- **Time estimate**: 45-90 minutes (8 CPU cores)

### Full Multi-Static Array (16 simulations)
- **Serial runtime**: 12-24 hours
- **Parallel runtime**: 1.5-3 hours (on HPC with 16-core array job)
- **Total storage**: ~1-2 GB (input + output files)

## Workflow

### 1. Generate Input Files
```bash
python generate_realistic_brain_inputs.py
```
Creates 16 input files in `brain_monopole_realistic/`

### 2. Run Simulations (HPC)
```bash
sbatch run_monopole_brain_array.sh
```
Or manually for each:
```bash
gprmax brain_monopole_realistic/brain_realistic_tx01.in -n 8
```

### 3. Extract S-Parameters
```bash
python extract_port_sparameters.py
```
Reads V(t) and I(t) from all 16 .out files, computes S-matrix, saves `brain_monopole_calibrated.s16p`

### 4. Visualize Results
```bash
python visualize_s16p.py brain_monopole_calibrated.s16p
```

## Comparison: Old vs. New Model

| Parameter | Original (Hertzian) | Realistic (Monopole) |
|-----------|---------------------|----------------------|
| Frequency range | 0-130 GHz (Nyquist) | 0-2 GHz (targeted) |
| Antenna type | Hertzian dipole | Quarter-wave monopole |
| Antenna length | N/A (point source) | 37.5 mm |
| Ground plane | None | 75×75 mm PEC |
| Port type | E-field probe (#rx) | 50Ω transmission line |
| S-parameter method | E/H approximation | Z = V/I, proper |
| Coupling medium | None | 5 mm tissue-equivalent |
| Head model | Concentric spheres (r=90,80,60mm) | Spheres from ellipsoid avg (r~95mm) |
| Domain size | 0.5×0.5×0.5 m | 0.6×0.6×0.6 m |
| Total cells | 15.6 M | 27.0 M |
| Time window | 10 ns | 15 ns |
| Runtime/sim | 20-40 min | 45-90 min |

## Next Steps

### Short-term (Before HPC Run)
- [x] Generate realistic input files
- [ ] Test single simulation locally (verify convergence)
- [ ] Validate geometry (visual check with geometry_view)
- [ ] Upload to HPC
- [ ] Submit array job

### Medium-term (After First Results)
- [ ] Extract S-parameters from 16 simulations
- [ ] Generate heatmaps and frequency plots
- [ ] Identify hemorrhage signature in S-matrix
- [ ] Compare with Hertzian dipole results

### Long-term (Optional Enhancements)
- [ ] Implement true ellipsoid geometry (custom Python mesh)
- [ ] Add more realistic skull (bone layers with different properties)
- [ ] Include cerebrospinal fluid (CSF) layer
- [ ] Test different hemorrhage sizes and positions
- [ ] Optimize antenna positions using pattern search
- [ ] Apply machine learning for hemorrhage detection

## References

1. Gabriel, C., Gabriel, S., & Corthout, E. (1996). The dielectric properties of biological tissues. Physics in Medicine & Biology, 41(11), 2231.
2. Mobashsher, A. T., & Abbosh, A. M. (2016). Artificial human phantoms: Human proxy in testing microwave apparatuses that have electromagnetic interaction with the human body. IEEE Microwave Magazine, 17(6), 42-62.
3. Persson, M., et al. (2014). Microwave-based stroke diagnosis making global prehospital thrombolytic treatment possible. IEEE Transactions on Biomedical Engineering, 61(11), 2806-2817.

## Troubleshooting

### Issue: Simulation crashes / Out of memory
**Solution**: Reduce domain size or increase dx_dy_dz (coarser mesh)

### Issue: Poor S-parameter quality (noisy, unrealistic)
**Solution**: 
- Increase time window (better frequency resolution)
- Check antenna port positions (must be on PEC boundary)
- Verify material properties (no typos in permittivity/conductivity)

### Issue: No hemorrhage signature detected
**Solution**:
- Verify lesion position (should be within sensitivity range)
- Try larger lesion size
- Increase frequency (better spatial resolution, but less penetration)
- Use difference imaging (with-lesion minus without-lesion)

### Issue: Long runtime on HPC
**Solution**:
- Use GPU acceleration (if available: `-gpu`)
- Optimize CPU parallelization (`-n` flag)
- Request high-memory nodes
- Consider domain decomposition for very large models

## Contact & Support

For questions or issues:
1. Check gprMax documentation: https://docs.gprmax.com
2. Review this README and MONOPOLE_HPC_README.md
3. Verify input file syntax with simple test cases
4. Check HPC logs for specific error messages

---
*Document version: 1.0*  
*Last updated: 2025*  
*Generated for thesis work on microwave brain hemorrhage imaging*

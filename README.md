# Spin2Scaling

## **📝 Credit**
- **Original Git Repository:** [SpaceTime (Spin1)](https://github.com/markus-kivioja/SpaceTime/tree/spin1)
- **Related Paper:** [GPU-accelerated DEC-based Gross-Pitaevskii solver](https://www.sciencedirect.com/science/article/pii/S0010465522001461?via%3Dihub)

## **📌 Overview**
This project extends the GPU-accelerated **Gross-Pitaevskii equation solver** using **Discrete Exterior Calculus (DEC)**, implementing **grid scaling for Spin-2** based on the original Spin-2 code with a Body-Centric-Cubic (BCC) grid.

## **Simulation Setup Notes**
- **Release from Trap:**
  The system is released from the trap at `TOTAL_HOLD_TIME`.
- **Scaling Behavior:**
  Scaling occurs at every timestep (`dt = 1e-5` or smaller) due to **oscillations at the Larmor frequency**.

## **🛠 Configuration Changes**

### ** Equal Trap Frequencies**
Modify `Spin2GpeBCC.cu` (**Lines 116-117**):
```cpp
constexpr double trapFreq_r = 136.22; // Previous: 126
constexpr double trapFreq_z = 136.22; // Previous: 166
```
Need to run `COMPUTE_GROUND_STATE =1` to get `equal_ground_state_psi_20.00_112.00.dat`
Move `lambdas_equal.h5` into build for calculating expansion
Replace `std::string k_castin_dum = "lambdas_equal.h5"` in (**Line 1894**)

- **Radial Imaging**
- In KnotRamps.h change
const std::vector<double3> Bbs = { make_double3(0, 0, 0.219), make_double3(0, 0, 0), make_double3(0, 0, 0), make_double3(0, 3.0, 0) }; the last one move the 3.0 to the imaging component
const std::vector<double> BbDurations = { STATE_PREP_DURATION, CREATION_RAMP_DURATION, TOTAL_HOLD_TIME, 100 };
const std::vector<RampType> BbTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, Ramp
- in Spin2GpeBCC.cu use the correct basis #define BASIS Y_QUANTIZED line 8



## Citation
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
- family-names: "Kivioja"
  given-names: "Markus"
- family-names: "Räbinä"
  given-names: "Jukka"
title: "GPU-accelerated DEC-based Gross-Pitaevskii solver"
version: 1.0-alpha
doi: 10.5281/zenodo.5700296
date-released: 2021-11-14
url: "https://github.com/markus-kivioja/GpuDecGpe"

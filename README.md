# Modeling and Optimization of Twist Wave Propagation and Bursting in Vortex Dynamics Using Neural Operators

## Overview
This repository contains code for comparing **Fourier Neural Operators (FNOs)** with traditional **numerical solvers** for vortex evolution modeling. The objective is to evaluate whether FNOs can provide computational speed-up while maintaining accuracy in vortex dynamics simulations.

## Features
- Implementation of **Remeshed Vortex Method** for solving the **incompressible Navier–Stokes equations** in vorticity–velocity form.
- **Fourier Neural Operator (FNO)** implementation for learning vortex evolution patterns.
- Performance comparison using **Mean Squared Error (MSE)** and **Structural Similarity Index (SSIM)**.
- **Computational efficiency benchmarking** of FNO against numerical solvers.

## Repository Structure
```
|
|-- generate_fno_predictions.py    # Generate predictions using trained FNO model
|-- main.py                        # Numerical solver for vortex evolution
|-- main_no.py                     # Training script for FNO model
|-- measure_numerical_runtime.py    # Measure execution time of numerical solver
|-- plot_comparison_runtime.py      # Visualize runtime comparison
|-- analyze_fno_vs_numerical.py     # Analyze MSE & SSIM between FNO and numerical method
|-- compare_runtime.py              # Compare execution time of FNO vs numerical solver
|-- vortex_data.npy                 # Preprocessed simulation data (not included in repo)
|-- vortex_fno.pth                  # Trained FNO model weights (not included in repo)
```

## Installation
To run the code, ensure you have Python 3.x installed along with the required dependencies:

```bash
pip install numpy torch scipy matplotlib pandas tqdm scikit-image
```

## Usage
### Running Numerical Simulations
```bash
python main.py
```

### Measuring Numerical Solver Execution Time
```bash
python measure_numerical_runtime.py
```

### Training Fourier Neural Operator
```bash
python main_no.py
```

### Generating FNO Predictions
```bash
python generate_fno_predictions.py
```

### Evaluating Model Performance
```bash
python analyze_fno_vs_numerical.py
```

### Comparing Execution Times
```bash
python compare_runtime.py
```

### Plotting Runtime Comparisons
```bash
python plot_comparison_runtime.py
```

## Results
- **MSE Analysis:** Shows FNO captures large-scale vortex structures but loses fine-scale details over time.
- **SSIM Analysis:** Demonstrates that FNO maintains structural similarity but struggles with turbulence preservation.
- **Computational Efficiency:** FNO achieves up to **267× speed-up** compared to traditional numerical solvers.

## References
For more details, refer to the paper:
> *"Modeling and Optimization of Twist Wave Propagation and Bursting in Vortex Dynamics Using Neural Operators"* by Ansh Tiwari.

## License
This project is released under the MIT License.

## Contact
For any questions, please contact **anshtiwari9899@gmail.com**.


# Modeling Twist Wave Propagation and Bursting in Vortex Dynamics Using Fourier Neural Operators (FNOs)

## Overview
This repository contains implementations of Fourier Neural Operators (FNOs) and a numerical solver to model the evolution of vorticity in a straight vortex tube. The goal is to compare the accuracy and efficiency of FNOs against traditional numerical methods and determine whether neural operators can predict vortex evolution faster while maintaining accuracy.

The FNO model is trained to predict the evolution of the vorticity field and energy dynamics in the axisymmetric r-z plane.

## Features
- **Numerical Solver**: Uses a remeshed vortex method to solve the Navier–Stokes equations
- **Fourier Neural Operator (FNO)**: Learns a mesh-invariant mapping from vorticity fields to future states
- **Speedup Comparison**: The FNO is up to 267× faster than traditional solvers
- **Energy Evolution Modeling**: Predicts the evolution of energy over time
- **GIF Visualization**: Compare numerical vs. FNO-predicted vorticity evolution

## Repository Structure
```
|
|-- generate_fno_predictions.py      # Generate predictions using FNO model
|-- main.py                         # Numerical solver
|-- train_fno.py                    # Train the Fourier Neural Operator
|-- analyze_results.py              # Compare numerical and FNO performance
|-- requirements.txt                # Python dependencies
|-- README.md                       # This documentation
```

## Installation
Ensure you have Python 3.x installed. Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
Run the numerical solver:
```bash
python main.py
```

Train the Fourier Neural Operator:
```bash
python train_fno.py
```

Generate FNO predictions:
```bash
python generate_fno_predictions.py
```

Analyze model performance:
```bash
python analyze_results.py
```

## 1. Numerical Method for Data Generation
The dataset is generated using a remeshed vortex method to solve the incompressible Navier–Stokes equations in vorticity–velocity formulation:

$\frac{\partial \boldsymbol{\omega}}{\partial t} + (\mathbf{u} \cdot \nabla) \boldsymbol{\omega} = (\boldsymbol{\omega} \cdot \nabla) \mathbf{u} + \nu \nabla^2 \boldsymbol{\omega}$

where:
- $\boldsymbol{\omega} = \nabla \times \mathbf{u}$ is the vorticity field
- $\mathbf{u}$ is the velocity field obtained by solving:
  
  $\nabla^2 \mathbf{u} = -\nabla \times \boldsymbol{\omega}$
- $\nu$ is the kinematic viscosity

### Initial Vorticity Field
The initial vorticity field for a straight vortex tube with an axially varying core size is given by:

$\omega(r,z,0) = \frac{\Gamma_0}{\pi \sigma(z)^2} \exp\left( -\frac{r^2}{\sigma(z)^2} \right)$

where $\sigma(z)$ encodes the axial perturbation:

$\sigma(z) = \sigma_{\min} + \frac{1}{2}(\sigma_{\max} - \sigma_{\min}) \left[1 + \sin\left(\frac{2\pi}{\lambda} z \right) \right]$

- Different perturbation profiles include sinusoidal, cosine, Gaussian, and none
- The remeshed vortex method evolves the vorticity field over time using fourth-order Runge–Kutta (RK4) integration
- Snapshots of the vorticity field are saved at fixed time intervals (every 0.5s)

## 2. Fourier Neural Operator (FNO) Model
The Fourier Neural Operator (FNO) is an advanced neural network designed to learn mappings between infinite-dimensional function spaces. Unlike CNNs, which perform local convolutions, the FNO learns global dependencies using the Fourier transform.

### Mathematical Formulation
Given an input function $u(r, z, t)$, the FNO models the operator:

$\mathcal{G}_\theta: u(r,z,t) \mapsto u(r,z,t + \Delta t)$

### Architecture
The FNO model consists of the following components:

1. **Lifting Layer**  
   Maps the input function to a high-dimensional latent space:
   $v_0(r,z) = P(u(r,z,t))$
   where $P$ is a pointwise fully connected layer.

2. **Fourier Layers**  
   The core of the model consists of $T$ Fourier layers, each applying:
   $v_{t+1}(r,z) = \sigma \left( W v_t(r,z) + \mathcal{F}^{-1} \left( R \cdot \mathcal{F} ( v_t(r,z) ) \right) \right)$
   - $\mathcal{F}$ and $\mathcal{F}^{-1}$ are the Fourier and inverse Fourier transforms
   - $R$ is a learnable weight tensor applied to the lowest $n_{\text{modes}}$ Fourier coefficients
   - $W$ is a pointwise linear transformation

3. **Projection Layer**  
   Projects the output of the last Fourier layer back to the physical space:
   $\hat{u}(r,z,t+\Delta t) = Q(v_T(r,z))$
   where $Q$ is a fully connected layer.

### Final Learned Operator
The full learned operator is:
$\mathcal{G}_\theta ( u(r,z,t) ) = Q \circ \left[ \sigma \left( W v_t(r,z) + \mathcal{F}^{-1} \left( R \cdot \mathcal{F} ( v_t(r,z) ) \right) \right) \right]_{t=0}^{T-1} \circ P(u(r,z,t))$

### Why Use FNO for Vortex Dynamics?
- **Resolution Invariance**: Can predict at different grid resolutions
- **Global Convolution**: Captures long-range dependencies in the vorticity field
- **Computational Speed**: Faster than solving PDEs numerically

## 3. Results
### Vorticity Evolution (Numerical vs. FNO)
Below is a GIF showing the vorticity evolution over time, comparing numerical and FNO predictions.

<img src="https://github.com/user-attachments/assets/94e59df1-7062-41b7-93c8-763b701520de" alt="vorticity_evolution" loop=infinite />

## 4. Performance Metrics
To evaluate accuracy, we use:
1. **Mean Squared Error (MSE)**:
   $\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left( u_{\text{true}} - u_{\text{FNO}} \right)^2$
2. **Structural Similarity Index (SSIM)**:  
   Measures perceptual similarity between the numerical and FNO outputs.

### Speed Comparison
- Numerical Solver: 216.69s per simulation
- FNO Inference: 0.81s per simulation (267× faster!)

## Conclusion
- The FNO model successfully predicts vortex evolution with high accuracy
- It achieves a 267× speedup over the numerical solver
- The trained model can generalize to unseen initial conditions

Future Work: Extend to higher Reynolds numbers, include more complex vortex interactions, and explore hybrid ML-PDE solvers.

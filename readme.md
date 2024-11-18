# Quantum Network Simulation Package

A Python-based simulation package designed for modeling quantum networking experiments using beam-splitters. This package was originally developed to simulate experiments for the Silicon-Vacancy (SiV) team.

## Overview

This simulation toolkit enables researchers to model and analyze quantum optical networks, with a particular focus on beam-splitter-based experiments. It provides tools for simulating quantum state propagation through optical fiber networks and analyzing the resulting quantum states.

## Features

- Beam-splitter network simulation
- Optical fiber transmission modeling
- Support for Silicon-Vacancy center experiments
- Quantum state propagation calculations
- Network topology configuration
- Blind quantum computing simulations

## Installation

### Prerequisites

- Python 3.7 or higher
- Git

### Dependencies

The following Python packages are required:
- NumPy: For numerical computations
- SciPy: For scientific computations and optimization
- QuTiP: For quantum mechanics calculations
- NetworkX: For network topology handling
- Matplotlib: For visualization
- h5py: For data storage
- pandas: For data manipulation
- tqdm: For progress bars

### Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/AzizaSuleyman/simnet
```

2. Navigate to the project directory:

```bash
cd simnet
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

### SimulationCode Directory

The `SimulationCode` directory contains the core simulation components:

#### fiber_network.py
- Core simulation engine for quantum optical networks
- Beam-splitter modeling functions
- Fiber transmission calculations
- Network topology management
- Quantum state propagation algorithms

#### SiVnodes.py
- Silicon-Vacancy center modeling
- Quantum state manipulation
- Contrast calculations and optimization
- Cavity reflection calculations

#### SiVgates.py
- Quantum gate implementations
- Gate operation simulations
- Error modeling and analysis

#### BlindGatesSimulation.py
- Blind quantum computing protocols
- Security analysis tools
- Client-server interaction simulation
- Fidelity calculations

#### Plots.py
- Visualization utilities
- Data plotting functions
- Analysis result presentation

## Usage Examples

### Basic Network Simulation
```python
from SimulationCode.fiber_network import NetworkSimulator

# Create a new network
sim = NetworkSimulator()

# Add components
sim.add_beamsplitter(position=(0, 0))
sim.add_detector(position=(1, 0))

# Run simulation
results = sim.run()
```

### Creating an SiV Experiment
```python
from SimulationCode.fiber_network import FiberNetwork
from SimulationCode.SiVnodes import SiV

# Create SiV instance
siv = SiV(kappa_in=74.9e3, kappa_w=54.5e3, g=5.6e3)

# Create network
network = FiberNetwork(siv)


## Data Storage

Simulation results can be stored in CSV format in the `OutputFiles` directory, organized by experiment type and date. The file naming convention includes relevant parameters and timestamps for easy identification.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Contact

For questions and support, please open an issue in the repository.



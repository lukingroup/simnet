# Quantum Network Simulation Package

![example figure](Notebooks/FiguresJupyter/DoubleColumn_SubFig2.pdf)


A Python-based simulation package designed for modeling quantum networking experiments using beam-splitters. This package was originally developed to simulate experiments for the Silicon-Vacancy (SiV) team.

## Overview

This simulation toolkit enables simulating entangelemnt-based experiments for various multi-node quantum network topologies. It is designed for cavity-QED nodes with reflection based gates using weak coherent sources as photonic time-bins to build qubits and qudits. Each interaction step on the path of the time-bin is encoded as a beam splitter hamiltonian, which works in our case since number of ecitations is < 1. It provides tools for simulating quantum state propagation through optical fiber networks and analyzing the resulting quantum states.

## Features

- Beam-splitter operator based network simulation
- Support for Silicon-Vacancy center experiments
- Quantum state propagation calculations
- Network topology configuration
- Blind quantum computing simulations
- Serial and parallel entangelment protocols using photonic qubits and qudits (e.g. d = 4)

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
- defines the network geometry
- specifies all of the losses in the network: fiber coupling, frequency conversion, fiber loss, detection loss
- specify if entanglement geometry is parallel vs serial (the default)

#### SiVnodes.py
- Silicon-Vacancy center class 
- Defines the cavity-QED parameters for SiV-cavity system
- Optimum SiV-Cavity detuning for hgherst contrast
- Change and set contrast
- Defines complex reflectivity of the cavity-SiV system at a function of optical frequency

#### SiVgates.py
- Contains a myriads of frequently used functions 
- State tomography
- Microwave and RF gates
- Photon loss
- Beam splitter operator definition

#### BlindGatesSimulation.py
- Contains the Blind quantum computing class 
- All of the function used to simulate Blind gates in our exeriment

#### Plots.py
- Visualization utilities
- Data plotting functions
- Analysis result presentation

## Usage Examples

### Create 2 node experiment with 2 SiVs

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import qutip as qt
import sys
sys.path.append('../SimulationCode/')
from TelescopeSimulation import *
from fiber_network import FiberNetwork
from SiVnodes import SiV
from SiVgates import *
from Plots import *

#Server A
siv_a = SiV(kappa_in= (74.9 - 54.5)*(10**3), kappa_w= (54.5)*(10**3), g=5.6*(10**3), wCav = (0)*(10**3), 
             wSiv = -(479.8 -639.6)*(10**3), dwEl = 0.5*(10**3))
#Server B
siv_b = SiV(kappa_in= (43.5 - 26.0)*(10**3), kappa_w= (26.0)*(10**3), g=8.5*(10**3), wCav = (0)*(10**3), 
             wSiv = -(804.9 -657.6)*(10**3), dwEl = -0.5*(10**3))

[Blind gates experiment](Notebooks/Blind_SingleNode_dataMatch.ipynb)


### Creating an SiV Experiment


## Data Storage

Simulation results can be stored in CSV format in the `OutputFiles` directory, organized by experiment type and date. The file naming convention includes relevant parameters and timestamps for easy identification.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.



## Contact

For questions and support, please open an issue in the repository.



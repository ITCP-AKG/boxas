# BO-XAS: Bayesian Optimization for X-ray Absorption Spectra

BO-XAS is a Python package designed to tune parameters for FEFF and FDMNES simulations of X-ray Absorption Near Edge Structure (XANES) spectra. It leverages Bayesian Optimization to efficiently find the best simulation parameters that match experimental data.

This project utilizes several powerful libraries:
- **Optuna**: For implementing the Bayesian Optimization algorithm.
- **ASE (Atomic Simulation Environment)**: For creating and manipulating crystal structures.
- **xasproc**: For preprocessing, normalizing, and analyzing XAS spectra.

## Features

- **Automated Parameter Tuning**: Tune parameters for both FEFF and FDMNES simulations.
- **Efficient Optimization**: Uses Bayesian Optimization (via Optuna) to minimize the number of expensive simulations required.
- **Flexible Objective Functions**: Supports various metrics to quantify the difference between simulated and experimental spectra (e.g., L2 distance, cosine similarity).
- **Structure Modeling**: Integrates with ASE for programmatic generation of atomic input structures.
- **Data Processing**: Leverages `xasproc` for robust handling of XAS data, including energy alignment, normalization, and interpolation.
- **Result Visualization**: Includes utilities to plot the comparison between the best-fit simulated spectrum and the experimental data.
- **Reproducibility**: The workflow is managed through configuration files and Jupyter notebooks, ensuring that results can be easily reproduced.

## Installation

The package can be installed directly from the GitHub repository using `uv` or `pip`.

```bash
uv pip install git+https://github.com/ITCP-AKG/boxas.git
```

Or with `pip`:

```bash
pip install git+https://github.com/ITCP-AKG/boxas.git
```

## Dependencies

- Python 3.8+
- [Optuna](https://optuna.org/)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [xasproc](https://github.com/ITCP-AKG/xasproc)
- A running MySQL/MariaDB server for Optuna's study storage.

You will need to set up a database for Optuna. The connection string is configured in the source code (e.g., `mysql+pymysql://optuna:optuna_pw@localhost/optuna_db`).

## Workflow

The typical workflow involves three main steps: configuration, optimization, and analysis. These steps are demonstrated in the provided Jupyter notebooks.

### 1. Configuration

All settings for the project are managed through a central YAML configuration file (e.g., `cfg_foils.yaml`). This file defines:
- **Project Directories**: Paths to the project root, output directories, etc.
- **Experimental Data**: Information about the experimental spectra, including file paths, energy ranges for normalization, and edge energies.
- **Simulation Parameters**: The parameter space for the Bayesian Optimization.

### 2. Run Optimization

The optimization process is launched from a script or notebook (e.g., `notebooks/j01-optimize-feff.ipynb`).
- It sets up an Optuna `study`.
- An `objective` function is defined, which takes a set of parameters, runs a simulation (FEFF or FDMNES), preprocesses the resulting spectrum, compares it to the experimental data using a chosen metric, and returns a score.
- Optuna's optimizer explores the parameter space to minimize this score.
- Each simulation run is stored in a unique directory for later inspection.

### 3. Analyze and Visualize Results

After the optimization is complete, the results can be analyzed.
- **Plotting Spectra (`notebooks/j02-plot-spectra.ipynb`)**: This notebook loads the best trial from the Optuna study, retrieves the corresponding simulated spectrum, and plots it against the experimental data for visual comparison.
- **Generating Tables (`notebooks/j03-build-tables.ipynb`)**: This notebook extracts the optimized parameters and objective function values for the best trials and formats them into LaTeX tables for inclusion in publications.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or suggestions.
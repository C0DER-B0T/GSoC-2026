# GSoC 2026: Benchmarking Framework for Causal Discovery Methods

This repository contains the prototype for the pgmpy benchmarking framework.

## Repository Structure
- `benchmarking/`: Core framework implementation.
- `run_benchmark.py`: Main entry point for running the benchmark.
- `results/`: Directory for storing output results (CSV, JSON, Pickle).
- `setup_cloud.sh`: Shell script for setting up the environment in a cloud platform (e.g., Colab).
- `requirements.txt`: Python dependencies.

## How to Run on Cloud (Colab/T4 GPU)
1. Clone this repository in your cloud environment:
   ```bash
   git clone https://github.com/C0DER-B0T/GSoC-2026.git
   cd GSoC-2026
   ```
2. Run the setup script:
   ```bash
   bash setup_cloud.sh
   ```
3. Run the benchmark:
   ```bash
   python run_benchmark.py
   ```

## Output
The benchmark results (CSV, JSON, and Pickle for models) will be saved in the `results/` folder with a timestamp. You can then push these results back to the repository.

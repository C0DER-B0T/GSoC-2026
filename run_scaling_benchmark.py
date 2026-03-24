import os
import sys
import pandas as pd
from datetime import datetime
from benchmarking import (
    BenchmarkRunner, 
    LinearGaussianSimulator,
    PCMethod, 
    GESMethod, 
    SHD,
    Precision,
    Recall,
    F1Score
)

def main():
    # 1. Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"scaling_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # 2. Define Scaling Benchmark
    # We will test how algorithms scale from 5 to 30 nodes
    node_counts = [5, 10, 20, 30]
    simulators = [
        LinearGaussianSimulator(n_nodes=n, n_samples=1000, edge_density=0.2)
        for n in node_counts
    ]

    methods = [
        PCMethod(ci_test="pearsonr"),
        GESMethod(score="bic")
    ]

    metrics = [
        SHD(),
        Precision(),
        Recall(),
        F1Score()
    ]

    # 3. Initialize Runner
    runner = BenchmarkRunner(
        simulators=simulators,
        methods=methods,
        metrics=metrics,
        n_runs=5, # 5 runs for each configuration for statistical stability
        random_state=42
    )

    # 4. Run Benchmark
    print(f"Starting scaling benchmark (Nodes: {node_counts})...")
    print("This may take some time depending on the number of nodes.")
    result, models = runner.run()

    # 5. Save Results
    result.summary()
    result.to_csv(os.path.join(results_dir, "scaling_results.csv"))
    result.to_json(os.path.join(results_dir, "scaling_metadata.json"))
    result.save_models(models, os.path.join(results_dir, "scaling_models.pkl"))

    print(f"\nScaling benchmark completed successfully! Results are saved in {results_dir}")

if __name__ == "__main__":
    main()

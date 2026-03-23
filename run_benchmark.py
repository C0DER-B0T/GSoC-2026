import os
import sys
import pandas as pd
from datetime import datetime
from benchmarking import (
    BenchmarkRunner, 
    ExampleModelSimulator, 
    PCMethod, 
    GESMethod, 
    SHD
)

def main():
    # 1. Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # 2. Define Benchmark Components
    simulators = [
        ExampleModelSimulator(model_name="alarm", n_samples=500),
        # Add more simulators if needed
    ]

    methods = [
        PCMethod(ci_test="chi_square"),
        # GESMethod() # Can add if needed
    ]

    metrics = [
        SHD(),
        # Add more metrics if needed
    ]

    # 3. Initialize Runner
    runner = BenchmarkRunner(
        simulators=simulators,
        methods=methods,
        metrics=metrics,
        n_runs=3,
        random_state=42
    )

    # 4. Run Benchmark
    print("Starting benchmark run...")
    result, models = runner.run()

    # 5. Save Results
    result.summary()
    result.to_csv(os.path.join(results_dir, "benchmark_results.csv"))
    result.to_json(os.path.join(results_dir, "benchmark_metadata.json"))
    result.save_models(models, os.path.join(results_dir, "benchmark_models.pkl"))

    print(f"\nBenchmark completed successfully! Results are saved in {results_dir}")

if __name__ == "__main__":
    main()

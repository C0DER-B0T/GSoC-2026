import time
import numpy as np
import pandas as pd
from datetime import datetime
from .result import BenchmarkResult

class BenchmarkRunner:
    """
    Orchestrates the benchmarking of multiple simulators, methods, and metrics.
    """
    def __init__(self, simulators, methods, metrics, n_runs=1, random_state=42):
        self.simulators = simulators
        self.methods = methods
        self.metrics = metrics
        self.n_runs = n_runs
        self.random_state = random_state

    def run(self):
        """Runs the benchmark across all configurations."""
        results = []
        models = {} # Store true and pred DAGs

        for sim in self.simulators:
            for i in range(self.n_runs):
                seed = self.random_state + i
                true_dag, data = sim.simulate(seed=seed)

                for method in self.methods:
                    start_time = time.time()
                    pred_dag = method.fit(data)
                    elapsed_time = time.time() - start_time

                    run_results = {
                        "Simulator": sim.name,
                        "Method": method.name,
                        "Run": i + 1,
                        "Runtime (s)": elapsed_time
                    }

                    for metric in self.metrics:
                        if metric.name != "Runtime (s)":
                            score = metric.compute(true_dag, pred_dag)
                            run_results[metric.name] = score

                    results.append(run_results)
                    models[f"{sim.name}_{method.name}_run_{i+1}"] = {
                        "true_dag": true_dag,
                        "pred_dag": pred_dag
                    }

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "n_runs": self.n_runs,
            "random_state": self.random_state,
            "simulators": [sim.name for sim in self.simulators],
            "methods": [method.name for method in self.methods],
            "metrics": [metric.name for metric in self.metrics]
        }

        return BenchmarkResult(pd.DataFrame(results), metadata=metadata), models

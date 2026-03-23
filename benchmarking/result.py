import pandas as pd
import json
import pickle
import os

class BenchmarkResult:
    """
    Immutable result object for benchmarking runs.
    """
    def __init__(self, data: pd.DataFrame, metadata: dict = None):
        self._data = data
        self._metadata = metadata or {}

    def summary(self):
        """Prints a summary table of the results."""
        summary_df = self._data.groupby(['Simulator', 'Method']).agg(['mean', 'std'])
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(summary_df)
        print("="*50)

    def to_csv(self, path: str):
        """Saves results to a CSV file."""
        self._data.to_csv(path, index=False)
        print(f"Results saved to {path}")

    def to_json(self, path: str):
        """Saves results and metadata to a JSON file."""
        result_dict = {
            "data": self._data.to_dict(orient="records"),
            "metadata": self._metadata
        }
        with open(path, 'w') as f:
            json.dump(result_dict, f, indent=4)
        print(f"Results and metadata saved to {path}")

    def save_models(self, models: dict, path: str):
        """Saves models using pickle."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(models, f)
        print(f"Models saved to {path}")

    @property
    def dataframe(self):
        return self._data

    @property
    def metadata(self):
        return self._metadata

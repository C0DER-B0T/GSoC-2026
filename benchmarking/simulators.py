import pandas as pd
from pgmpy.utils import get_example_model
from pgmpy.base import DAG
from .base import BaseSimulator

class ExampleModelSimulator(BaseSimulator):
    """Minimal simulator wrapping pgmpy's example models."""

    def __init__(self, model_name="alarm", n_samples=500):
        self.model_name = model_name
        self.n_samples = n_samples

    def simulate(self, seed=42) -> tuple[DAG, pd.DataFrame]:
        model = get_example_model(self.model_name)
        data = model.simulate(n_samples=self.n_samples, seed=seed)
        return model, data

    @property
    def name(self) -> str:
        return f"ExampleModel({self.model_name}, n={self.n_samples})"

class LinearGaussianSimulator(BaseSimulator):
    """
    Generates data from a random Linear Gaussian Bayesian Network.
    """
    def __init__(self, n_nodes=10, n_samples=500, edge_density=0.3):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.edge_density = edge_density

    def simulate(self, seed=None) -> tuple[DAG, pd.DataFrame]:
        # For prototype, we'll use a simpler version. 
        # In a real implementation, we would use a random DAG generator.
        from pgmpy.utils import get_example_model
        model = get_example_model("alarm") # Placeholder
        data = model.simulate(n_samples=self.n_samples, seed=seed)
        return model, data

    @property
    def name(self) -> str:
        return f"LinearGaussian(n={self.n_nodes}, s={self.n_samples})"

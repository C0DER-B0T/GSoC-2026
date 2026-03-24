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
    def __init__(self, n_nodes=10, n_samples=500, edge_density=0.2, noise_std=1.0):
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.edge_density = edge_density
        self.noise_std = noise_std

    def simulate(self, seed=None) -> tuple[DAG, pd.DataFrame]:
        import numpy as np
        from pgmpy.models import BayesianNetwork
        from pgmpy.factors.discrete import TabularCPD
        from pgmpy.base import DAG
        
        if seed is not None:
            np.random.seed(seed)

        # 1. Generate a random DAG
        # Simple random DAG generation: 
        # For each pair (i, j) where i < j, add an edge with probability edge_density
        nodes = [f'X{i}' for i in range(self.n_nodes)]
        dag = DAG()
        dag.add_nodes_from(nodes)
        
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                if np.random.rand() < self.edge_density:
                    dag.add_edge(nodes[i], nodes[j])

        # 2. Convert to BayesianNetwork to simulate data
        # For Linear Gaussian, we simulate values directly based on the DAG structure
        # X_i = sum(w_ji * X_j) + epsilon_i
        data = pd.DataFrame(index=range(self.n_samples), columns=nodes)
        
        # Sort nodes topologically to ensure parents are simulated before children
        import networkx as nx
        topo_order = list(nx.topological_sort(dag))
        
        # Random weights between 0.5 and 2.0
        weights = {edge: np.random.uniform(0.5, 2.0) for edge in dag.edges()}
        
        for node in topo_order:
            parents = list(dag.get_parents(node))
            noise = np.random.normal(0, self.noise_std, size=self.n_samples)
            if not parents:
                data[node] = noise
            else:
                val = np.zeros(self.n_samples)
                for p in parents:
                    val += weights[(p, node)] * data[p]
                data[node] = val + noise
                
        return dag, data

    @property
    def name(self) -> str:
        return f"LinearGaussian(n={self.n_nodes}, s={self.n_samples}, d={self.edge_density})"

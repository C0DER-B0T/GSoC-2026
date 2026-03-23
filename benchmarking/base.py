from abc import ABC, abstractmethod
import pandas as pd
from pgmpy.base import DAG

class BaseSimulator(ABC):
    """Abstract base for all data simulators."""

    @abstractmethod
    def simulate(self, seed: int = None) -> tuple[DAG, pd.DataFrame]:
        """
        Returns
        -------
        true_dag : DAG
            The ground-truth causal structure.
        data : pd.DataFrame
            Simulated observational data.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for results table."""
        pass


class BaseMethod(ABC):
    """Abstract base for all causal discovery method wrappers."""

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> DAG:
        """
        Learn a causal structure from data.

        Returns
        -------
        estimated_dag : DAG
            The estimated causal graph.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class BaseMetric(ABC):
    """Abstract base for all evaluation metrics."""

    @abstractmethod
    def compute(self, true_dag: DAG, pred_dag: DAG) -> float:
        """Compute metric score. Lower or higher = better depends on metric."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def higher_is_better(self) -> bool:
        return True  # Override in SHD (lower is better)

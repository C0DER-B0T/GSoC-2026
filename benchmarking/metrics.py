from pgmpy.base import DAG
from .base import BaseMetric

class SHD(BaseMetric):
    """
    Structural Hamming Distance.
    """
    def compute(self, true_dag: DAG, pred_dag: DAG) -> float:
        true_edges = set(true_dag.edges())
        pred_edges = set(pred_dag.edges())
        # Count missing, extra, and reversed edges
        missing  = len(true_edges - pred_edges)
        extra    = len(pred_edges - true_edges)
        reversed_edges = sum(
            1 for (u, v) in true_edges
            if (v, u) in pred_edges
        )
        return float(missing + extra + reversed_edges)

    @property
    def name(self): return "SHD"

    @property
    def higher_is_better(self): return False

class Precision(BaseMetric):
    """
    Edge Precision.
    """
    def compute(self, true_dag, pred_dag) -> float:
        true_edges = set(true_dag.edges())
        pred_edges = set(pred_dag.edges())
        if not pred_edges:
            return 0.0
        return len(true_edges.intersection(pred_edges)) / len(pred_edges)

    @property
    def name(self): return "Precision"

class Recall(BaseMetric):
    """
    Edge Recall.
    """
    def compute(self, true_dag, pred_dag) -> float:
        true_edges = set(true_dag.edges())
        pred_edges = set(pred_dag.edges())
        if not true_edges:
            return 1.0
        return len(true_edges.intersection(pred_edges)) / len(true_edges)

    @property
    def name(self): return "Recall"

class F1Score(BaseMetric):
    """
    Edge F1-Score.
    """
    def compute(self, true_dag, pred_dag) -> float:
        prec = Precision().compute(true_dag, pred_dag)
        rec = Recall().compute(true_dag, pred_dag)
        if (prec + rec) == 0:
            return 0.0
        return 2 * (prec * rec) / (prec + rec)

    @property
    def name(self): return "F1-Score"

class Runtime(BaseMetric):
    """
    Execution time metric.
    """
    def __init__(self, elapsed_time=0.0):
        self.elapsed_time = elapsed_time

    def compute(self, true_dag, pred_dag) -> float:
        return self.elapsed_time

    @property
    def name(self): return "Runtime (s)"

    @property
    def higher_is_better(self): return False

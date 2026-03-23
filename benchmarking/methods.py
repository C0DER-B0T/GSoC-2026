import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import PC, GES
from .base import BaseMethod

class PCMethod(BaseMethod):
    """
    Wrapper for pgmpy's PC algorithm.
    """
    def __init__(self, ci_test="pearsonr", alpha=0.05):
        self.ci_test = ci_test
        self.alpha = alpha

    def fit(self, data: pd.DataFrame) -> DAG:
        pc = PC(data=data)
        return pc.estimate(
            ci_test=self.ci_test,
            significance_level=self.alpha,
            return_type="dag",
            show_progress=False
        )

    @property
    def name(self) -> str:
        return f"PC({self.ci_test})"

class GESMethod(BaseMethod):
    """
    Wrapper for pgmpy's GES algorithm.
    """
    def __init__(self, score="bic"):
        self.score = score

    def fit(self, data: pd.DataFrame) -> DAG:
        ges = GES(data=data)
        return ges.estimate(
            score=self.score,
            show_progress=False
        )

    @property
    def name(self) -> str:
        return f"GES({self.score})"

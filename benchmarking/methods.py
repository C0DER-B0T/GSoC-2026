import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import PC
from pgmpy.causal_discovery import GES
from pgmpy.estimators.CITests import pearsonr, chi_square
from pgmpy.estimators.scoring import BicScore, BDeuScore, K2Score
from .base import BaseMethod

class PCMethod(BaseMethod):
    """
    Wrapper for pgmpy's PC algorithm.
    """
    def __init__(self, ci_test="pearsonr", alpha=0.05):
        self.ci_test_str = ci_test
        self.alpha = alpha

    def fit(self, data: pd.DataFrame) -> DAG:
        if self.ci_test_str == "pearsonr":
            ci_func = pearsonr
        elif self.ci_test_str == "chi_square":
            ci_func = chi_square
        else:
            raise ValueError(f"Unsupported ci_test: {self.ci_test_str}")

        pc = PC(data=data)
        return pc.estimate(
            variant='stable',
            ci_test=ci_func,
            significance_level=self.alpha,
            show_progress=False
        )

    @property
    def name(self) -> str:
        return f"PC({self.ci_test_str})"

class GESMethod(BaseMethod):
    """
    Wrapper for pgmpy's GES algorithm.
    """
    def __init__(self, score="bic"):
        self.score = score

    def fit(self, data: pd.DataFrame) -> DAG:
        if self.score.lower() == "bic":
            score_obj = BicScore(data)
        elif self.score.lower() == "bdeu":
            score_obj = BDeuScore(data)
        elif self.score.lower() == "k2":
            score_obj = K2Score(data)
        else:
            raise ValueError(f"Unsupported score: {self.score}")

        ges = GES(data)
        return ges.estimate(scoring_method=score_obj, show_progress=False)

    @property
    def name(self) -> str:
        return f"GES({self.score})"

import pandas as pd
import logging
from pgmpy.base import DAG
from .base import BaseMethod

# Helper for robust imports across different pgmpy versions
def _import_pgmpy_item(module_paths, item_name):
    import importlib
    for path in module_paths:
        try:
            module = importlib.import_module(path)
            return getattr(module, item_name)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Could not find {item_name} in any of {module_paths}")

# Robustly import required pgmpy classes
try:
    PC = _import_pgmpy_item(['pgmpy.estimators', 'pgmpy.estimators.PC'], 'PC')
    GES = _import_pgmpy_item(['pgmpy.estimators', 'pgmpy.causal_discovery', 'pgmpy.estimators.GES'], 'GES')
    
    # CI Tests
    pearsonr = _import_pgmpy_item(['pgmpy.estimators.CITests', 'pgmpy.estimators'], 'pearsonr')
    chi_square = _import_pgmpy_item(['pgmpy.estimators.CITests', 'pgmpy.estimators'], 'chi_square')
    
    # Scoring methods
    BicScore = _import_pgmpy_item(['pgmpy.estimators', 'pgmpy.estimators.scoring', 'pgmpy.scoring'], 'BicScore')
    BDeuScore = _import_pgmpy_item(['pgmpy.estimators', 'pgmpy.estimators.scoring', 'pgmpy.scoring'], 'BDeuScore')
    K2Score = _import_pgmpy_item(['pgmpy.estimators', 'pgmpy.estimators.scoring', 'pgmpy.scoring'], 'K2Score')
except ImportError as e:
    # Fallback for logging if pgmpy isn't fully set up yet
    print(f"Warning: {e}")

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
        # Try different parameter names as they vary between dev versions
        try:
            return pc.estimate(
                variant='stable',
                ci_test=ci_func,
                significance_level=self.alpha,
                show_progress=False
            )
        except TypeError:
            # Fallback for older/different parameter names
            return pc.estimate(
                ci_test=ci_func,
                significance_level=self.alpha,
                return_type='dag',
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
        try:
            return ges.estimate(scoring_method=score_obj, show_progress=False)
        except TypeError:
            # Fallback for different parameter names (some use 'score')
            return ges.estimate(score=score_obj, show_progress=False)

    @property
    def name(self) -> str:
        return f"GES({self.score})"

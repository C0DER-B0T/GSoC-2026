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
def _safe_import(paths, name):
    try:
        return _import_pgmpy_item(paths, name)
    except ImportError:
        return None

PC = _safe_import(['pgmpy.causal_discovery', 'pgmpy.estimators', 'pgmpy.estimators.PC'], 'PC')
GES = _safe_import(['pgmpy.causal_discovery', 'pgmpy.estimators', 'pgmpy.estimators.GES'], 'GES')

# CI Tests - trying new naming convention from warnings
pearsonr = _safe_import(['pgmpy.ci_tests', 'pgmpy.estimators.CITests', 'pgmpy.estimators'], 'pearsonr')
if pearsonr is None:
    pearsonr = _safe_import(['pgmpy.ci_tests'], 'Pearsonr')

chi_square = _safe_import(['pgmpy.ci_tests', 'pgmpy.estimators.CITests', 'pgmpy.estimators'], 'chi_square')
if chi_square is None:
    chi_square = _safe_import(['pgmpy.ci_tests'], 'ChiSquare')

# Scoring methods
BicScore = _safe_import(['pgmpy.estimators.scoring', 'pgmpy.estimators', 'pgmpy.scoring'], 'BicScore')
BDeuScore = _safe_import(['pgmpy.estimators.scoring', 'pgmpy.estimators', 'pgmpy.scoring'], 'BDeuScore')
K2Score = _safe_import(['pgmpy.estimators.scoring', 'pgmpy.estimators', 'pgmpy.scoring'], 'K2Score')

# Final verification
missing = [name for name, val in [('PC', PC), ('GES', GES), ('pearsonr', pearsonr), ('BicScore', BicScore)] if val is None]
if missing:
    print(f"Warning: The following pgmpy components could not be found: {', '.join(missing)}")

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

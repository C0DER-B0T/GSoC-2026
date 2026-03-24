import pandas as pd
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

# Try to import from the new causal_discovery module first (dev branch)
PC_new = _safe_import(['pgmpy.causal_discovery'], 'PC')
GES_new = _safe_import(['pgmpy.causal_discovery'], 'GES')

# Fallback to the old estimators module
PC_old = _safe_import(['pgmpy.estimators'], 'PC')
GES_old = _safe_import(['pgmpy.estimators'], 'GES')

class PCMethod(BaseMethod):
    """
    Wrapper for pgmpy's PC algorithm.
    """
    def __init__(self, ci_test="pearsonr", alpha=0.05, variant="parallel"):
        self.ci_test_str = ci_test
        self.alpha = alpha
        self.variant = variant

    def fit(self, data: pd.DataFrame) -> DAG:
        # Use the new API if available
        if PC_new is not None:
            import inspect
            sig = inspect.signature(PC_new.__init__)
            params = sig.parameters
            
            # Build arguments based on what __init__ accepts
            kwargs = {}
            if 'variant' in params: kwargs['variant'] = self.variant
            if 'ci_test' in params: kwargs['ci_test'] = self.ci_test_str
            if 'significance_level' in params: kwargs['significance_level'] = self.alpha
            if 'show_progress' in params: kwargs['show_progress'] = False
            if 'return_type' in params: kwargs['return_type'] = 'dag'
            
            # Check if it's the new class-based PC (doesn't take data in __init__)
            if 'data' not in params:
                pc = PC_new(**kwargs)
                pc.fit(data)
                return pc.causal_graph_

        # Fallback to old API
        PC_to_use = PC_new if PC_new is not None else PC_old
        if PC_to_use is None:
            raise ImportError("Could not find PC algorithm in pgmpy.")
            
        pc = PC_to_use(data=data)
        return pc.estimate(
            ci_test=self.ci_test_str,
            significance_level=self.alpha,
            return_type="dag",
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
        # Determine the correct scoring string for the new API
        from pgmpy.utils import get_dataset_type
        dtype = get_dataset_type(data)
        
        score_mapping = {
            "continuous": {"bic": "bic-g", "aic": "aic-g", "ll": "ll-g"},
            "discrete": {"bic": "bic-d", "aic": "aic-d", "k2": "k2", "bdeu": "bdeu"},
            "mixed": {"bic": "bic-cg", "aic": "aic-cg", "ll": "ll-cg"}
        }
        
        score_str = score_mapping.get(dtype, {}).get(self.score.lower(), self.score.lower())

        # Use the new API if available
        if GES_new is not None:
            import inspect
            sig = inspect.signature(GES_new.__init__)
            params = sig.parameters
            
            # Build arguments based on what __init__ accepts
            kwargs = {}
            if 'scoring_method' in params: kwargs['scoring_method'] = score_str
            if 'return_type' in params: kwargs['return_type'] = 'dag'
            # GES dev version doesn't seem to have show_progress in __init__
            if 'show_progress' in params: kwargs['show_progress'] = False
            
            if 'data' not in params:
                ges = GES_new(**kwargs)
                ges.fit(data)
                return ges.causal_graph_

        # Fallback to old API
        GES_to_use = GES_new if GES_new is not None else GES_old
        if GES_to_use is None:
            raise ImportError("Could not find GES algorithm in pgmpy.")

        # For old API, we might need the score object
        try:
            from pgmpy.estimators import BicScore, BDeuScore, K2Score
            scores = {"bic": BicScore, "bdeu": BDeuScore, "k2": K2Score}
            score_class = scores.get(self.score.lower())
            if score_class:
                score_obj = score_class(data)
                ges = GES_to_use(data=data, score=score_obj)
                return ges.estimate(show_progress=False)
        except ImportError:
            pass
            
        # Last resort fallback
        ges = GES_to_use(data=data)
        return ges.estimate(score=self.score, show_progress=False)

    @property
    def name(self) -> str:
        return f"GES({self.score})"

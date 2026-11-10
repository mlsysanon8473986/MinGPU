"""MinGPU: from a target loss to GPUs, parallel plan and model size.

MinGPU calls Calculon's Python API in-process to evaluate parallel plans and
writes every search to a local DuckDB database, so no loose files are produced.
"""

__version__ = "0.1.0"

from .planner.define_search_space.candidate_models import Model
from .planner.define_search_space.request import SearchRequest
from .planner.define_search_space.scaling_law import (
    BESIROGLU,
    CHINCHILLA,
    ScalingLaw,
    UnreachableLoss,
)
from .planner.define_search_space.search_space import SearchSpace
from .planner.execute_search.inner_search import (
    InnerSearchResult,
    PlanCounts,
    WorkerPool,
    count_plans,
    inner_search,
)
from .planner.execute_search.outer_search import PairProgress, run_outer_search
from .planner.objectives.result import ObjectiveResult, run_objective
from .planner.objectives.rules import Answer
from .planner.workflow import run_request
from .storage.db import DEFAULT_DB_PATH, connect, migrate

__all__ = [
    "__version__",
    "DEFAULT_DB_PATH",
    "connect",
    "migrate",
    # the model family and the loss law
    "Model",
    "BESIROGLU",
    "CHINCHILLA",
    "ScalingLaw",
    "UnreachableLoss",
    # the search of one pair and of a whole planner search
    "PlanCounts",
    "InnerSearchResult",
    "WorkerPool",
    "count_plans",
    "inner_search",
    "PairProgress",
    "SearchSpace",
    "run_outer_search",
    # the request of the paper and its answer
    "SearchRequest",
    "run_request",
    "run_objective",
    "ObjectiveResult",
    "Answer",
]

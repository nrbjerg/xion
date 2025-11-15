from xion.types import Solution
from xion.core.branch_and_bound import branch_and_bound
from xion.models.milp import MILP
from typing import Optional, Callable
from xion.utils.results import convert_solver_result_to_MILP_result

def solve(problem: MILP, time_budget: Optional[float] = None, verbose: bool = False) -> Optional[Solution]:
    """Solves the supplied MILP hopefully within the time budget."""
    sol = branch_and_bound(problem, time_budget=time_budget, verbose=verbose)
    return None if sol is None else convert_solver_result_to_MILP_result(sol, problem)



    

        
        
        
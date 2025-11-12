from xion.types import Vector
from xion.core.branch_and_bound import branch_and_bound
from xion.models.milp import MILP
from typing import Tuple
from xion.utils.results import convert_solver_result_to_MILP_result

def solve(problem: MILP, time_budget: float = 0.0) -> Tuple[float, Vector]:
    """Solves the supplied MILP hopefully within the time budget."""
    sol = branch_and_bound(problem)
    return convert_solver_result_to_MILP_result(sol, problem)



    

        
        
        
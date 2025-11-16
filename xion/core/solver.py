from xion.types import Solution
#from xion.core.branch_and_bound_strong_branching_version import branch_and_bound
from xion.core.branch_and_bound import branch_and_bound
from xion.models.milp import MILP
from typing import Optional, Callable
from xion.utils.results import convert_solver_result_to_MILP_result
from xion.core.presolver import RecoveryPipeline
from xion.models.canonical_form import CanonicalForm

def solve(problem: MILP, time_budget: float = 0.0, verbose: bool = False) -> Optional[Solution]:
    """Solves the supplied MILP hopefully within the time budget."""
    canonical_form = CanonicalForm.from_milp(problem) 
    sol = branch_and_bound(canonical_form)
    return canonical_form.convert_solution_to_milp_solution(sol, problem) if sol != None else None



    

        
        
        
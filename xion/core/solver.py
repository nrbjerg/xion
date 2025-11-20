from xion.types import Solution
#from xion.core.branch_and_bound_strong_branching_version import branch_and_bound
from xion.core.branch_and_bound import branch_and_bound
from xion.models.milp import MILP
from typing import Optional, Callable
from xion.utils.results import convert_solver_result_to_MILP_result
from xion.models.canonical_form import CanonicalForm
from xion.core.presolver import root_presolve

def solve(problem: MILP, time_budget: Optional[float] = None, verbose: bool = False) -> Optional[Solution]:
    """Solves the supplied MILP hopefully within the time budget."""
    #presolved_canonical_form = CanonicalForm.from_milp(problem)
    if (root_presolve_res := root_presolve(CanonicalForm.from_milp(problem))) != None:
        presolved_canonical_form, recovery_pipeline = root_presolve_res
        sol = branch_and_bound(presolved_canonical_form, time_budget=time_budget, verbose=verbose)
        return recovery_pipeline.convert_sol_of_presolved_problem_to_MILP_solution(sol, problem) if sol != None else None
    else:
        return None



    

        
        
        
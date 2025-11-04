import scipy.optimize as opt
import numpy as np
from xion.types import Matrix, Vector
from xion.models.canonical_form import CanonicalForm
from xion.core.branch_and_bound import branch_and_bound
from xion.models.milp import MILP
from typing import List, Tuple

def solve(problem: MILP, time_budget: float = 0.0) -> Tuple[float, Vector]:
    """Solves the supplied MILP hopefully within the time budget."""
    canonical_form = CanonicalForm.from_milp(problem) 
    sol = branch_and_bound(canonical_form)
    return canonical_form.convert_solution_to_milp_solution(sol, problem)



    

        
        
        
import scipy.optimize as opt
import numpy as np
from xion.types import Matrix, Vector
from xion.models.canonical_form import CanonicalForm
from typing import List, Tuple

def branch_and_bound(problem: CanonicalForm) -> Tuple[float, Vector]:
    """Implements a simple branch and bound algorithm for solving a canonical MILP, 
       i.e. a problem as described in xion.models.canonical_form.py"""
    best_sol = None
    best_obj_val = np.inf

    # TODO: Prioritize the bounds based on their coefficients in the objective function somehow.
    stack: List[Tuple[Vector, Vector]] = [(problem.l, problem.u)]
    nodes_evaluated = 0

    while stack:
        nodes_evaluated += 1
        bounds_at_node = stack.pop()
        
        # Solve the LP relaxation of the problem
        res = opt.linprog(problem.c, A_eq = problem.A, b_eq = problem.b, bounds = np.vstack([bounds_at_node[0].T, bounds_at_node[1].T]).T)

        # Check for infeasibility and if so simply skip the value or if the value of the LP relaxation
        # is higher than the current best feasible value we may skip it.
        if not res.success or res.fun >= best_obj_val:
            continue
        
        for i in problem.integral_indices:
            # If one of the values which should be integral is not integral we branch.
            if not np.isclose(res.x[i], np.round(res.x[i])):
                val = res.x[i]

                new_l = bounds_at_node[0].copy()
                new_l[i] = np.ceil(val)
                stack.append((new_l, bounds_at_node[1]))

                new_u = bounds_at_node[1].copy()
                new_u[i] = np.floor(val)
                stack.append((bounds_at_node[0], new_u))
                break

        # NOTE: We have already checked if res.fun >= best_obj_val, so we now know that the solution is the current best.
        else: 
            best_sol = res.x
            best_obj_val = res.fun
            continue

    return best_obj_val, best_sol
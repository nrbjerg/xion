import time 
import scipy.optimize as opt
import numpy as np
from xion.types import Matrix, Vector
from xion.models.canonical_form import CanonicalForm
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import heapq
from loguru import logger
from numba import njit

@njit()
def compute_down_fractionality(lp_var_ass: Vector, integrality_mask: Vector) -> Vector:
    """Computes the down fractionally of the variables which should be integral in lp_var_ass."""
    integral_var_ass = lp_var_ass[integrality_mask]
    return integral_var_ass - np.floor(integral_var_ass)

@dataclass(order=True)
class Node: 
    """Stores all of the information needed in a branch and bound node."""
    lp_obj_val: float
    lp_var_ass: Vector = field(compare=False)
    bounds: Matrix = field(compare=False)

def branch_and_bound(problem: CanonicalForm, k: int = 4, time_budget: Optional[float] = None, verbose: bool = False) -> Optional[Tuple[float, Vector]]:
    """Implements a simple branch and bound algorithm for solving a canonical MILP, 
       i.e. a problem as described in xion.models.canonical_form.py, additionally it uses
       best-bound node selection, strong-branching (TODO) and pseudo-costs (TODO)"""
    N = problem.c.shape[0]
    best_sol = None
    best_obj_val = np.inf

    pseudo_costs = np.zeros((N, 2), dtype=np.double)
    strong_branches = np.zeros(N, dtype=np.double)

    # Solve the original LP-relaxation to the MILP, and add it to the set of open nodes.
    original_bounds = np.column_stack([problem.lb, problem.ub])
    res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds=original_bounds)

    # NOTE: If the original LP relaxation to the MILP problem is infeasible, then the MILP is likewise infeasible
    if not res.success: 
        if verbose:
            logger.info(f"The MILP problem {problem.identifier} was infeasible as the root LP-relaxation was infeasible")

        return None 

    open_nodes: List[Node] = [Node(lp_obj_val=res.fun, lp_var_ass=np.array(res.x), bounds=original_bounds)]
    heapq.heapify(open_nodes)

    nodes_evaluated = 0
    start_time = time.perf_counter()
    while len(open_nodes) != 0 and ((time_budget is None) or (time.perf_counter() - start_time <= time_budget)):
        nodes_evaluated += 1
        node: Node = heapq.heappop(open_nodes)

        down_fractionality = compute_down_fractionally()
        
        for i in problem.integral_indices:
            # If one of the values which should be integral is not integral we branch.
            if not np.isclose(node.lp_sol[i], np.round(node.lp_sol[i])):
                for j in range(2):
                    # Solve the LP relaxation of the branched sub problems
                    bounds = np.copy(node.bounds)
                    if j == 0:
                        bounds[i, 0] = np.ceil(node.lp_sol[i])
                    else:
                        bounds[i, 1] = np.floor(node.lp_sol[i])

                    res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds = bounds)

                    # Performs pruning by checking if the LP-relaxation was solvable (otherwise the MILP subproblem is not)
                    # and by checking if the minima for the relaxation was lower than the best objective value (of the MILP)
                    if res.success and res.fun <= best_obj_val:
                        heapq.heappush(open_nodes, Node(lp_obj=res.fun, lp_sol=res.x, bounds = bounds))

                break

        # NOTE: We have already checked if res.fun >= best_obj_val, so we now know that the solution 
        #       is the current best. (as we use best-node selection) 
        else:
            if verbose:
                logger.success(f"Branch and bound evaluated {nodes_evaluated} nodes in {time.perf_counter() - start_time:.1f} seconds.")

            return node.lp_obj, node.lp_sol

    return best_obj_val, best_sol
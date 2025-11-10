import scipy.optimize as opt
import numpy as np
from xion.types import Matrix, Vector
from xion.models.canonical_form import CanonicalForm
from typing import List, Tuple, Optional
from dataclasses import dataclass
import heapq

@dataclass(order=True)
class Node: 
    """Stores all of the information needed in a branch and bound node."""
    lp_obj: float
    lp_sol: Vector
    bounds: Matrix

def branch_and_bound(problem: CanonicalForm) -> Optional[Tuple[float, Vector]]:
    """Implements a simple branch and bound algorithm for solving a canonical MILP, 
       i.e. a problem as described in xion.models.canonical_form.py, additionally it uses
       best-bound node selection, strong-branching (TODO) and """
    best_sol = None
    best_obj_val = np.inf

    # Solve the original LP-relaxation to the MILP, and add it to the set of open nodes.
    original_bounds = np.vstack([problem.l, problem.u]).T
    print(original_bounds.shape)
    res = opt.linprog(problem.c, A_eq=problem.A, b_eq=problem.b, bounds=original_bounds)
    if not res.success: 
        # NOTE: If the original LP relaxation to the MILP problem is infeasible, then the MILP is likewise infeasible
        return None 
    open_nodes: List[Node] = [Node(lp_obj=res.fun, lp_sol=np.array(res.x), bounds=original_bounds)]
    heapq.heapify(open_nodes)

    nodes_evaluated = 0
    while len(open_nodes) != 0:
        nodes_evaluated += 1
        node: Node = heapq.heappop(open_nodes)
        
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

                    res = opt.linprog(problem.c, A_eq = problem.A, b_eq = problem.b, bounds = bounds)

                    # Performs pruning by checking if the LP-relaxation was solvable (otherwise the MILP subproblem is not)
                    # and by checking if the minima for the relaxation was lower than the best objective value (of the MILP)
                    if res.success and res.fun <= best_obj_val:
                        heapq.heappush(open_nodes, Node(lp_obj=res.fun, lp_sol=res.x, bounds = bounds))

                break

        # NOTE: We have already checked if res.fun >= best_obj_val, so we now know that the solution is the current best.
        else: 
            best_sol = node.lp_sol
            best_obj_val = node.lp_obj 
            continue

    return best_obj_val, best_sol
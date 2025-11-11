import scipy.optimize as opt
import numpy as np
from xion.types import Matrix, Vector
from xion.models.canonical_form import CanonicalForm
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import heapq
from numba import njit

@njit()
def is_integral(lp_sol: Vector, integrality_mask: Vector) -> bool:
    """Checks if the solution (lp_sol) to the LP-relaxation, is a solution to the MILP."""
    return np.all(np.isclose(lp_sol[integrality_mask], np.round(lp_sol[integrality_mask]), atol=1e-6))

@njit()
def find_candidate_branching_vars(lp_sol: Vector, integral_indices: Vector, k: int) -> Vector:
    """Finds the set consisting of the first k candidate variables for branching."""
    fractionality = np.minimum(lp_sol[integral_indices] - np.floor(lp_sol[integral_indices]), 
                               np.ceil(lp_sol[integral_indices]) - lp_sol[integral_indices])

    candidates = integral_indices[np.argpartition(-fractionality, k - 1)[:k]]
    non_integer_candidates = candidates[~np.isclose(lp_sol[candidates], np.round(lp_sol[candidates]), atol=1e-6)]
    return non_integer_candidates

    #j = 0
    #for i in integer_indices:
    #    if not np.isclose(lp_sol[i], np.round(lp_sol[i]), atol=1e-6):
    #        # Mark var i as a candidate, for future branching.
    #        candidates[j] = i
    #        j += 1
    #        
    #        if candidates[-1] != -1:
    #            break
    
    #return candidates
            

@dataclass(order=True)
class Node: 
    """Stores all of the information needed in a branch and bound node."""
    lp_obj: float
    lp_sol: Vector = field(compare=False)
    bounds: List[Tuple[float, float]] = field(compare=False)

def branch_and_bound(problem: CanonicalForm, alpha: float = 0.5) -> Optional[Tuple[float, Vector]]:
    """Implements a simple branch and bound algorithm for solving a canonical MILP, 
       i.e. a problem as described in xion.models.canonical_form.py, additionally it uses
       best-bound node selection, strong-branching (TODO) and """
    best_sol = None
    best_obj_val = np.inf

    # Solve the original LP-relaxation to the MILP, and add it to the set of open nodes.
    original_bounds = [(l, u) for l, u in zip(problem.l, problem.u)]
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
    
        # NOTE: Since we are using best-bound node selection, the first solution found is indeed the best possible solution.
        if is_integral(node.lp_sol, problem.integral_mask):
            return node.lp_obj, node.lp_sol

        # Perform strong-branching on the found variables.
        strong_branching_best_nodes = []
        strong_branching_best_score = -np.inf 
        for var_idx in find_candidate_branching_vars(node.lp_sol, problem.integral_indices, k = 1):
            lower_branching_bounds, upper_branching_bounds = list(node.bounds), list(node.bounds)
            lower_branching_bounds[var_idx] = (node.bounds[var_idx][0], np.floor(node.lp_sol[var_idx]))
            upper_branching_bounds[var_idx] = (np.ceil(node.lp_sol[var_idx]), node.bounds[var_idx][1]) 

            # Compute LP-relaxations
            lower_res = opt.linprog(problem.c, A_eq = problem.A, b_eq = problem.b, bounds = lower_branching_bounds)
            upper_res = opt.linprog(problem.c, A_eq = problem.A, b_eq = problem.b, bounds = upper_branching_bounds)

            # NOTE: Check if the LP-relaxation is still feasible otherwise we know that we do not need to branch any 
            #       further down this branch, since we will at some point have to fix variable i, (as it cannot be fractional).
            #       Additionally in the case where only one is feasible we can prune half of the search tree, 
            #       by picking this variable, hence we set the value of the improvement to np.inf.
            lower_inc = np.inf if not lower_res.success else lower_res.fun - node.lp_obj
            upper_inc = np.inf if not upper_res.success else upper_res.fun - node.lp_obj

            if (score := (min(lower_inc, upper_inc) + alpha * max(lower_inc, upper_inc)) > strong_branching_best_score):
                strong_branching_best_score = score
                strong_branching_best_nodes = [Node(lp_obj=res.fun, lp_sol=np.array(res.x), bounds=branching_bounds) 
                                               for res, branching_bounds in zip([lower_res, upper_res], [lower_branching_bounds, upper_branching_bounds]) if res.success]

        # NOTE: If the list is empty then the node was infeasible, and hence it can have no descendants in the MILP search tree.
        for strong_branching_node in strong_branching_best_nodes:
            heapq.heappush(open_nodes, strong_branching_node)

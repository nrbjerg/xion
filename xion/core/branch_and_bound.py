from highspy import HighsBasis, HighsSolution, HighsModelStatus
import scipy.optimize as opt
import time
import numpy as np
from xion.types import Matrix, Vector
from xion.models.milp import MILP
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
import heapq
from numba import njit
from xion.core.solve_lp_relaxation import LPRelaxationSolver
from xion.models.HiGHS_model import convert_MILP_to_HiGHS_lp_relaxation
from xion.utils.results import convert_solver_result_to_MILP_result
from loguru import logger

@njit()
def is_integral(lp_sol: Vector, integrality_mask: Vector) -> bool:
    """Checks if the solution (lp_sol) to the LP-relaxation, is a solution to the MILP."""
    return np.all(np.isclose(lp_sol[integrality_mask], np.round(lp_sol[integrality_mask]), atol=1e-9))

@njit()
def find_candidate_branching_vars(lp_sol: Vector, integral_indices: Vector, k: int) -> Vector:
    """Finds the set consisting of the first k candidate variables for branching."""
    fractionality = np.minimum(lp_sol[integral_indices] - np.floor(lp_sol[integral_indices]), 
                               np.ceil(lp_sol[integral_indices]) - lp_sol[integral_indices])

    candidates = integral_indices[np.argpartition(-fractionality, k - 1)[:k]]
    non_integer_candidates = candidates[~np.isclose(lp_sol[candidates], np.round(lp_sol[candidates]), atol=1e-9)]
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
    status: HighsModelStatus = field(compare=False)
    lp_sol: Vector = field(compare=False)
    basis: Optional[HighsBasis] = field(compare=False)
    bounds: Matrix = field(compare=False)

def branch_and_bound(problem: MILP, time_budget: Optional[float] = None, alpha: float = 0.5, verbose: bool = False) -> Optional[Tuple[float, Vector]]:
    """Implements a simple branch and bound algorithm for solving a MILP, 
       i.e. a problem as described in xion.models.milp.py, additionally it uses
       best-bound node selection, strong-branching and warm-starting of the LP-relaxation solver."""
    best_var_ass = None
    best_obj_val = np.inf
    lp_relaxation = convert_MILP_to_HiGHS_lp_relaxation(problem)
    lp_relaxation_solver = LPRelaxationSolver()

    # Solve the original LP-relaxation to the MILP, and add it to the set of open nodes.
    root_bounds = np.column_stack([problem.get_lower_bounds(), problem.get_upper_bounds()]) 

    root_res = lp_relaxation_solver.solve(lp_relaxation, None) 
    # NOTE: If the original LP relaxation to the MILP problem is infeasible, then the MILP is likewise infeasible
    if root_res is None: 
        if verbose:
            logger.info(f"The MILP problem {problem.identifier} was infeasible as the root LP-relaxation was infeasible")
        return None 

    open_nodes: List[Node] = [Node(*root_res, bounds=root_bounds)]
    heapq.heapify(open_nodes)

    start_time = time.perf_counter()
    nodes_evaluated = 0
    while len(open_nodes) != 0 and ((time_budget is None) or (time.perf_counter() - start_time <= time_budget)):
        nodes_evaluated += 1
        node: Node = heapq.heappop(open_nodes)
    
        # NOTE: Since we are using best-bound node selection, the first solution found is indeed the best possible solution.
        if is_integral(node.lp_sol, problem.integral_mask):
            return (node.lp_obj, node.lp_sol)
        
        # Set bounds of the LP-relaxation, according to those of the node.
        lp_relaxation.col_lower_ = node.bounds[:, 0]
        lp_relaxation.col_upper_ = node.bounds[:, 1]

        # Perform strong-branching on the found variables.
        strong_branching_best_nodes = []
        strong_branching_best_score = -np.inf 

        for var_idx in find_candidate_branching_vars(node.lp_sol, problem.integral_indices, k = 1):
            # TODO: Do something to minimize the copying, do we really need to copy everything instead of simply modifying the node bounds.
            lower_branching_bounds, upper_branching_bounds = node.bounds.copy(), node.bounds.copy()
            lower_branching_bounds[var_idx] = (node.bounds[var_idx][0], np.floor(node.lp_sol[var_idx]))
            upper_branching_bounds[var_idx] = (np.ceil(node.lp_sol[var_idx]), node.bounds[var_idx][1]) 

            # Compute LP-relaxations
            lp_relaxation.col_lower_ = lower_branching_bounds[:, 0]
            lp_relaxation.col_upper_ = lower_branching_bounds[:, 1]
            lower_res = lp_relaxation_solver.solve(lp_relaxation, node.basis)
            lp_relaxation.col_lower_ = upper_branching_bounds[:, 0]
            lp_relaxation.col_upper_ = upper_branching_bounds[:, 1]
            upper_res = lp_relaxation_solver.solve(lp_relaxation, node.basis)

            # NOTE: Check if the LP-relaxation is still feasible otherwise we know that we do not need to branch any 
            #       further down this branch, since we will at some point have to fix variable i, (as it cannot be fractional).
            #       Additionally in the case where only one is feasible we can prune half of the search tree, 
            #       by picking this variable, hence we set the value of the improvement to np.inf.
            lower_inc = np.inf if lower_res == None else lower_res[0] - node.lp_obj
            upper_inc = np.inf if upper_res == None else upper_res[0] - node.lp_obj

            if (score := (min(lower_inc, upper_inc) + alpha * max(lower_inc, upper_inc)) > strong_branching_best_score):
                strong_branching_best_score = score
                strong_branching_best_nodes = [Node(*res, bounds=branching_bounds) 
                                               for res, branching_bounds in zip([lower_res, upper_res], [lower_branching_bounds, upper_branching_bounds]) if res != None]

        # NOTE: If the list is empty then the node was infeasible, and hence it can have no descendants in the MILP search tree.
        for strong_branching_node in strong_branching_best_nodes:
            heapq.heappush(open_nodes, strong_branching_node)

    if verbose:
        logger.info(f"Branch and bound evaluated {nodes_evaluated} nodes in {time.perf_counter() - start_time:.2f} seconds.")
    return None if best_obj_val == np.inf else (best_obj_val, best_var_ass)
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
from xion.core.presolver import domain_propagation

#@njit()
def is_integral(x: Vector, integral_mask: Vector) -> bool:
    """Checks if the solution (lp_sol) to the LP-relaxation, is a solution to the MILP."""
    y = x[integral_mask]
    return np.all(np.isclose(y, np.round(y), atol=1e-6))

#@njit()
def compute_down_fractionality(x: Vector, integral_indices: Vector) -> Vector:
    """Computes the down fractionally of the variables which should be integral in x."""
    y = x[integral_indices]
    return y - np.floor(y)

#@njit()
def find_candidate_branching_variables(x: Vector, integral_indices: Vector, k: int) -> Vector:
    """Find good candidate branching variables, for now simply pick the k most fractional."""
    y = x[integral_indices]
    fractionality = np.minimum(y - np.floor(y), np.ceil(y) - y)

    candidates = integral_indices[np.argpartition(-fractionality, k - 1)[:k]]
    non_integer_candidates = candidates[~np.isclose(x[candidates], np.round(x[candidates]), atol=1e-6)]
    return non_integer_candidates

@dataclass(order=True)
class Node: 
    """Stores all of the information needed in a branch and bound node."""
    z: float
    x: Vector = field(compare=False)
    bounds: Matrix = field(compare=False)

def branch_and_bound(problem: CanonicalForm, k_branching_candidates: int = 4, k_pseudo: int = 4, time_budget: Optional[float] = None, verbose: bool = False) -> Optional[Tuple[float, Vector]]:
    """Implements a simple branch and bound algorithm for solving a canonical MILP, 
       i.e. a problem as described in xion.models.canonical_form.py, additionally it uses
       best-bound node selection, strong-branching (TODO) and pseudo-costs (TODO)"""
    N = problem.c.shape[0]
    primal_bound = np.inf
    primal_incumbent = None

    pseudo_costs = np.zeros((N, 2), dtype=np.double)
    number_of_pseudo_cost_estimates = np.zeros(N, dtype=np.uint64)

    # Solve the original LP-relaxation to the MILP, and add it to the set of open nodes.
    original_bounds = np.column_stack([problem.lb, problem.ub])
    res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds=original_bounds)
    # NOTE: If the original LP relaxation to the MILP problem is infeasible, then the MILP is likewise infeasible
    if not res.success: 
        if verbose:
            logger.info(f"The MILP problem {problem.identifier} was infeasible as the root LP-relaxation was infeasible")

        return None 

    open_nodes: List[Node] = [Node(z=res.fun, x=np.array(res.x), bounds=original_bounds)]
    heapq.heapify(open_nodes)

    nodes_evaluated = 0
    start_time = time.perf_counter()
    while len(open_nodes) != 0 and ((time_budget is None) or (time.perf_counter() - start_time <= time_budget)):
        nodes_evaluated += 1
        node: Node = heapq.heappop(open_nodes)
        #if (bounds := domain_propagation(problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, problem.integral_mask, problem.lb, problem.ub)) != None:
        #    node.bounds = np.column_stack(bounds)
        #else:
        #    continue # NOTE: Node was found to be infeasible during the bounds checking.

        if is_integral(node.x, problem.integral_mask) and node.z < primal_bound:
            primal_bound = node.z
            primal_incumbent = node.x

        best_branching_var_idx = None
        score_of_best_branching_var_idx = -np.inf

        candidate_branching_vars = find_candidate_branching_variables(node.x, problem.integral_indices, k = k_branching_candidates)
        down_fractionality = compute_down_fractionality(node.x, candidate_branching_vars)
        for i, var_idx in enumerate(candidate_branching_vars):
            # If the value is not integral, then consider branching on the node, if there has been a sufficient 
            # number (k) of strong branches on the node, then utilize the pseudo costs, otherwise perform strong branching.
            if not np.isclose(node.x[var_idx], np.round(node.x[var_idx]), atol=1e-6):
                var_down_fractionality, var_up_fractionality = down_fractionality[i], 1 - down_fractionality[i]

                # Utilize strong branching on the variable.
                if number_of_pseudo_cost_estimates[var_idx] < k_pseudo:   
                    # Solve the LP relaxation of the branched sub problems
                    original_bounds_of_var = node.bounds[var_idx].copy()
                    node.bounds[var_idx][0] = np.ceil(node.x[var_idx])
                    up_res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds = node.bounds)
                    node.bounds[var_idx][0] = original_bounds_of_var[0] 
                    node.bounds[var_idx][1] = np.floor(node.x[var_idx])
                    down_res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds = node.bounds)
                    node.bounds[var_idx, 1] = original_bounds_of_var[1] 

                    # NOTE: in the case where one of the sub problems are infeasible we immediately branch 
                    # on this variable, since we can fix it - and hence get a smaller problem / search space.
                    if (not down_res.success) or (not up_res.success):
                        # NOTE: Only one of them can at most be feasible - hence we may reuse the nodes bounds.
                        if down_res.success:
                            node.bounds[var_idx, 1] = np.floor(node.x[var_idx])
                            heapq.heappush(open_nodes, Node(z=down_res.fun, x=down_res.x, bounds = node.bounds))
                        elif up_res.success:
                            node.bounds[var_idx, 0] = np.ceil(node.x[var_idx])
                            heapq.heappush(open_nodes, Node(z=up_res.fun, x=up_res.x, bounds = node.bounds))

                        # NOTE: We do not need to check other variables - and in particular we
                        #       do not need to run the else block after this for loop (where)
                        #       the found variable is branched on, by computing its LP-relaxation values.
                        break 
                        
                    # In the case where both where LP-feasible update their pseudo-costs accordingly.
                    delta_down, delta_up = node.z - down_res.fun, node.z - up_res.fun

                    # NOTE: The score of branching on this variable, will subsequently be computed using the exact 
                    #       delta_down and delta_up, computed to produce an estimate of the pseudo costs.
                    estimates_of_pseudo_costs = np.array([delta_down / var_down_fractionality, delta_up / var_up_fractionality])
                    pseudo_costs[var_idx] = (1 / (number_of_pseudo_cost_estimates[var_idx] + 1)) * (pseudo_costs[var_idx] * number_of_pseudo_cost_estimates[var_idx] + estimates_of_pseudo_costs)
                    number_of_pseudo_cost_estimates[var_idx] += 1

                # Utilize pseudo costs to estimate sub-problem lp-relaxation values.
                else:
                    delta_down, delta_up = pseudo_costs[var_idx, 0] * var_down_fractionality, pseudo_costs[var_idx, 1] * var_up_fractionality

                if (score := delta_down * delta_up) > score_of_best_branching_var_idx:
                    best_branching_var_idx = var_idx
                    score_of_best_branching_var_idx = score

        else: # NOTE: For-Else pattern (else block gets run if we did not break, in this case if we did not find a variable fixing which resulted in an infeasible subproblem)
            if best_branching_var_idx == None: 
                # We did not find a variable to branch on, hence we have found a feasible MILP solution - 
                # notice that this solution must be optimal, since best-node selection was applied.
                if verbose:
                    logger.success(f"Branch and bound evaluated {nodes_evaluated} nodes in {time.perf_counter() - start_time:.1f} seconds.")
                return node.z, node.x

            else:
                # We did find a variable to branch on, which we will now do:
                up_bounds = node.bounds.copy()
                up_bounds[best_branching_var_idx, 0] = np.ceil(node.x[best_branching_var_idx])
                up_res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds = up_bounds)

                if up_res.success and up_res.fun < primal_bound:
                    heapq.heappush(open_nodes, Node(z=up_res.fun, x=up_res.x, bounds = up_bounds))

                down_bounds = node.bounds # NOTE: might as well reuse.
                down_bounds[best_branching_var_idx, 1] = np.floor(node.x[best_branching_var_idx])
                down_res = opt.linprog(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, bounds = down_bounds)

                if down_res.success and down_res.fun < primal_bound:
                    heapq.heappush(open_nodes, Node(z=down_res.fun, x=down_res.x, bounds = down_bounds))

                # If we did not update the pseudo costs of var_idx already, then we do so now - using the results from the LP-relaxation.
                if number_of_pseudo_cost_estimates[var_idx] > k_pseudo + 1:
                    delta_down, delta_up = node.z - down_res.fun, node.z - up_res.fun

                    # NOTE: The score of branching on this variable, will subsequently be computed using the exact 
                    #       delta_down and delta_up, computed to produce an estimate of the pseudo costs.
                    estimates_of_pseudo_costs = np.array([delta_down / var_down_fractionality, delta_up / var_up_fractionality])
                    pseudo_costs[var_idx] = (1 / (number_of_pseudo_cost_estimates[var_idx] + 1)) * (pseudo_costs[var_idx] * number_of_pseudo_cost_estimates[var_idx] + estimates_of_pseudo_costs)
                    number_of_pseudo_cost_estimates[var_idx] += 1

    logger.info(f"Evaluated {nodes_evaluated} nodes at ({nodes_evaluated / (time.perf_counter() - start_time):.1f} nodes/s)")         
    return primal_bound, primal_incumbent
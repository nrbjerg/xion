from dataclasses import dataclass, field
from xion.types import Scalar, Matrix, Vector
from typing import Dict, Tuple, List, Optional
from xion.models.canonical_form import CanonicalForm
import numpy as np
from numba import njit, prange
from loguru import logger
import time 

#@njit()
def domain_propagation(A_ineq: Matrix, b_ineq: Matrix, A_eq: Matrix, b_eq: Vector, integral_mask: Vector, lb: Vector, ub: Vector) -> Optional[Tuple[Vector, Vector]]:
    """Perform domain propagation - i.e. try to see if any variables can be fixed given the lower and upper bounds, using the constraint
       matrices, i.e. lb[i] = ub[i] = 1 and x_i + x_j <= 1 => x_j = 0, returns None if the given bounds results in an infeasible problem."""
    M_ineq, N = A_ineq.shape
    M_eq = b_eq.shape[0]

    if not np.all(lb <= ub):
        return None
    
    tightened_lb, tightened_ub = lb.copy(), ub.copy()
    while True: # Keep looping until a fixed point is reached. 
        for a, b in zip(A_ineq, b_ineq):
            pos_mask = a > 0
            neg_mask = a < 0

            a_min = np.dot(a[pos_mask], lb[pos_mask]) + np.dot(a[neg_mask], ub[neg_mask])
            a_max = np.dot(a[pos_mask], ub[pos_mask]) + np.dot(a[neg_mask], lb[neg_mask])
            for i in prange(N):
                if a[i] > 0:
                    tightened_ub[i] = min(tightened_ub[i], (b - a_min + a[i] * lb[i]) / a[i])
                elif a[i] < 0:
                    tightened_lb[i] = max(tightened_lb[i], (b - a_max + a[i] * ub[i]) / a[i])
                    

        # TODO: Do equality as well.
                
        # NOTE: Check if the tightened bounds implies that the node is infeasible.
        if np.any(tightened_lb > tightened_ub): 
            return None 
        
        # Exploit the integral nature of some of the variables.
        tightened_lb[integral_mask] = np.ceil(tightened_lb[integral_mask])
        tightened_ub[integral_mask] = np.floor(tightened_ub[integral_mask])

        lb, ub = tightened_lb.copy(), tightened_ub.copy()
        # NOTE: In case we have reached a fix point, we can safely return 
        if np.all(tightened_lb == lb) and np.all(tightened_ub == ub):
            return tightened_lb, tightened_ub

# TODO: Implement actual presolving
def root_presolve(problem: CanonicalForm) -> Optional[CanonicalForm]:
    """Performs presolving on a MILP in canonical form, given the lower and upper bounds."""
    # Tighten bounds on variables - based on domain propagation.
    start_time = time.perf_counter()
    M_initial = problem.b_ineq.shape[0] + problem.b_eq.shape[0]
    if (bounds := domain_propagation(problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, problem.integral_indices, problem.lb, problem.ub)) != None:
        problem.lb = bounds[0] 
        problem.ub = bounds[1]
    else:
        logger.warning(f"Model was found to be infeasible during presolving in {time.perf_counter() - start_time:.3f}s.")
        return None

    # TODO: Check for redundant constraints / infeasibility using min and max activity.
    logger.success(f"Finished Presolving in {time.perf_counter() - start_time:.3f}s, removed {problem.b_ineq.shape[0] + problem.b_eq.shape[0] - M_initial} rows.")
    return problem


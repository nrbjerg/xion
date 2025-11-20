from dataclasses import dataclass, field
from xion.types import Scalar, Matrix, Vector
from xion.models.milp import MILP, Variable
from typing import Dict, Tuple, List, Optional
from xion.models.canonical_form import CanonicalForm
import numpy as np
from numba import njit, prange
from loguru import logger
from xion.constants import THETA, EPSILON, INF
import time 

@dataclass
class RecoveryPipeline:
    """Stores the information needed to reverse any transformation made by the presolver."""
    fixed_variables: Vector
    fixed_values: Vector

    def convert_sol_of_presolved_problem_to_MILP_solution(self, sol: Vector, problem: MILP) -> Tuple[float, Dict[Variable, Scalar]]:
        """Converts the solution to the presolved MILP to a solution to the actual MILP"""
        obj_val, vals_of_canonical_vars = sol
        if problem.obj_sense == "max":
            obj_val *= -1.0
        
        vals_of_milp_vars = np.empty(len(problem.vars))
        vals_of_milp_vars[self.fixed_variables] = self.fixed_values[self.fixed_variables]
        vals_of_milp_vars[~self.fixed_variables] = vals_of_canonical_vars

        var_val_mapping = {var: int(round(val)) if np.isclose(val, np.round(val)) else val 
                           for var, val in zip(problem.vars, vals_of_milp_vars)}

        return obj_val, var_val_mapping


def ineq_individual_row_presolving(A_ineq: Matrix, b_ineq: Vector, lb: Vector, ub: Vector, integral_mask: Vector) -> Optional[Tuple[bool, Matrix, Vector, Vector, Vector]]:
    """Performs presolving on individual rows of the inequality matrix, i.e. the matrix A_ineq such that A_ineq x <= b_ineq."""
    m, n = A_ineq.shape
    has_changed = False
    for i in range(m - 1, -1, -1): # TODO: See if it works better as a parallel loop (although then the sequential benefits will be absent.)
        pos_mask = A_ineq[i] > 0
        neg_mask = A_ineq[i] < 0

        min_activity = np.dot(A_ineq[i, pos_mask], lb[pos_mask]) + np.dot(A_ineq[i, neg_mask], ub[neg_mask])
        max_activity = np.dot(A_ineq[i, pos_mask], ub[pos_mask]) + np.dot(A_ineq[i, neg_mask], lb[neg_mask])

        # Check for infeasibility 
        if min_activity > b_ineq[i]:
            return None 

        # Check if the constraint is redundant. 
        if max_activity <= b_ineq[i] + EPSILON:
            b_ineq = np.delete(b_ineq, i, axis=0)
            A_ineq = np.delete(A_ineq, i, axis=0)
            has_changed = True
            continue
            
        # 2.1.2 Perform bounds strengthening on row i.
        if max_activity < INF:
            for j in range(n):
                if A_ineq[i, j] == 0:
                    continue

                ell_ij = np.dot(A_ineq[i, pos_mask], lb[pos_mask]) + np.dot(A_ineq[i, neg_mask], ub[neg_mask]) - A_ineq[i, j] * (lb[j] if A_ineq[i,j] > 0 else ub[j])
                rhs = (b_ineq[i] - ell_ij) / A_ineq[i,j]
                if A_ineq[i, j] > 0: 
                    potential_ub = np.floor(rhs) if integral_mask[j] else rhs 
                    if ub[j] - potential_ub > THETA:
                        ub[j] = potential_ub
                        has_changed = True
                
                else:
                    potential_lb = np.ceil(rhs) if integral_mask[j] else rhs 
                    if potential_lb - lb[j] > THETA:
                        lb[j] = potential_lb
                        has_changed = True
                        
        # 2.1.3 TODO: Perform coefficient strengthening.
    
    return has_changed, A_ineq, b_ineq, lb, ub

# TODO: Actually implement this:
def eq_individual_row_presolving(A_eq: Matrix, b_eq: Vector, lb: Vector, ub: Vector, integral_mask: Vector) -> Optional[Tuple[bool, Matrix, Vector, Vector, Vector]]: 
    """Performs presolving on individual rows of the equality matrix, i.e. the matrix A_eq such that A_eq x = b_eq."""
    return False, A_eq, b_eq, lb, ub

def root_presolve_loop(c: Vector, A_ineq: Matrix, b_ineq: Vector, A_eq: Matrix, b_eq: Vector, integral_mask: Vector, lb: Vector, ub: Vector) -> Optional[Tuple[Vector, float, Matrix, Vector, Matrix, Vector, Vector, Vector, Vector]]:
    """The main presolving loop for the root node of the problem."""
    cn = 0.0
    lb[integral_mask] = np.ceil(lb[integral_mask])
    ub[integral_mask] = np.floor(ub[integral_mask])

    has_changed = True
    while has_changed:
        n = A_ineq.shape[1]

        # 2.1 Individual row presolving on A_ineq
        has_changed_ineq, A_ineq, b_ineq, lb, ub = ineq_individual_row_presolving(A_ineq, b_ineq, lb, ub, integral_mask)
            
        # 2.2 Perform individual row presolving on A_eq.
        has_changed_eq, A_eq, b_eq, lb, ub = eq_individual_row_presolving(A_eq, b_eq, lb, ub, integral_mask)

        # 2.3 Perform individual variable reductions if applicable (if no changes was made, then no variables can be changed.)
        has_changed = has_changed_ineq or has_changed_eq

        if has_changed:
            # 2.3 Perform individual reductions on variables.
            for j in range(n - 1, -1, -1):
                # 2.3.1 Remove fixed variables from the problem:
                if np.isclose(lb[j], ub[j], EPSILON):
                    cn += c[j] * lb[j]
                    b_ineq += A_ineq[:, j] * lb[j]
                    b_eq += A_eq[:, j] * lb[j]

                    A_eq = np.delete(A_eq, j, axis=1)
                    A_ineq = np.delete(A_ineq, j, axis=1)
                    c = np.delete(c, j, axis = 0)
                    lb = np.delete(lb, j, axis = 0)
                    ub = np.delete(ub, j, axis = 0)
                    integral_mask = np.delete(integral_mask, j, axis=0)
                    
        #        # 2.3.2 Implied variable free substitution

    return c, cn, A_ineq, b_ineq, A_eq, b_eq, integral_mask, lb, ub

def root_presolve(problem: CanonicalForm) -> Optional[Tuple[CanonicalForm, RecoveryPipeline]]:
    """Performs presolving on a MILP in canonical form, given the lower and upper bounds."""
    # Tighten bounds on variables - based on domain propagation.
    start_time = time.perf_counter()
    M_initial = problem.b_ineq.shape[0] + problem.b_eq.shape[0]
    N_initial = problem.A_eq.shape[1]
    if (presolve_res := root_presolve_loop(problem.c, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, problem.integral_mask, problem.lb, problem.ub)) == None:
        logger.warning(f"Model was found to be infeasible during presolving in {time.perf_counter() - start_time:.3f}s.")
        return None
    else: 
        # Update model accordingly:
        problem.c, problem.cn, problem.A_ineq, problem.b_ineq, problem.A_eq, problem.b_eq, problem.integral_mask, problem.lb, problem.ub = presolve_res

        problem.integral_indices = np.array([i for i in range(problem.A_eq.shape[1]) if problem.integral_mask[i] == True])

    logger.success(f"Finished Presolving in {time.perf_counter() - start_time:.3f}s, removed {problem.b_ineq.shape[0] + problem.b_eq.shape[0] - M_initial} rows and {N_initial - problem.A_eq.shape[1]} variables")
    return problem, RecoveryPipeline(np.array([True, False, True, False]), np.zeros(4))


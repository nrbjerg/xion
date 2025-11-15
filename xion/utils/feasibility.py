from numba import njit
import numpy as np
from xion.types import Matrix, Vector, Scalar
from xion.models.milp import MILP, Variable
from typing import Dict, Optional   

@njit()
def is_integral(lp_sol: Vector, integrality_mask: Vector) -> bool:
    """Checks if the solution (lp_sol) to the LP-relaxation, is a solution to the MILP."""
    return np.all(np.isclose(lp_sol[integrality_mask], np.round(lp_sol[integrality_mask]), atol=1e-9))

@njit()
def is_lp_relaxation_feasible(var_ass: Vector, A: Matrix, b_leq: Vector, b_geq: Vector) -> bool:
    """Efficiently check wether the variable assignment is a solution to the LP-relaxation"""
    con_vals = A @ var_ass
    for i in range(A.shape[0]):
        if con_vals[i] <= b_leq[i] or con_vals[i] >= b_geq[i]:
            return False

    return True

def compute_number_of_violated_constraints(milp: MILP, var_ass: Dict[Variable, Scalar]) -> int:
    """Computes the number of violated constraints of the given variable assignment."""
    number_of_violations = 0
    for con in milp.cons:
        lhs = sum(con.lc.weights.get(var, 0.0) * var_ass[var] for var in milp.vars)
        # Check that the condition actually holds at the variable assignment
        if ((con.rel == "=" and lhs != con.rhs) or 
            (con.rel == "<=" and lhs > con.rhs) or
            (con.rel == ">=" and lhs < con.rhs)):    
            number_of_violations += 1

    return number_of_violations
from xion.models.milp import MILP, Variable
from xion.utils.generators.integer_generalized_assignment_problem import generate_IGAP
from xion.utils.results import convert_solver_result_to_MILP_result
from xion import solve
from typing import Tuple, Dict, Optional
from xion.types import Scalar

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
 
def evaluate_milp_at_var_ass(milp: MILP, var_ass: Dict[Variable, Scalar]) -> Optional[Scalar]:
    """Tries to evaluate the MILP at the given variable assignment, returns None if the variable assignment is infeasible"""
    if compute_number_of_violated_constraints(milp, var_ass) != 0:
        return None
    
    return sum(milp.obj_fun.weights.get(var, 0.0) * var_ass[var] for var in milp.vars)
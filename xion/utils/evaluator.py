from xion.models.milp import MILP, Variable
from xion.utils.generators.integer_generalized_assignment_problem import generate_IGAP
from xion.utils.results import convert_solver_result_to_MILP_result
from xion import solve
from typing import Tuple, Dict, Optional
from xion.types import Scalar, Vector, Matrix
 
def evaluate_milp_at_var_ass(milp: MILP, var_ass: Dict[Variable, Scalar]) -> Scalar:
    """Tries to evaluate the MILP at the given variable assignment, returns None if the variable assignment is infeasible"""
    return sum(milp.obj_fun.weights.get(var, 0.0) * var_ass[var] for var in milp.vars)
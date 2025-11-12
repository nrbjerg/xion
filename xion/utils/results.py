from typing import Tuple, Dict
from xion.models.milp import MILP
from xion.types import Vector, Scalar
import numpy as np

def convert_solver_result_to_MILP_result(sol: Tuple[float, Vector], problem: MILP) -> Tuple[float, Dict[str, Scalar]]:
    """Converts the solver results to the a human friendly format (i.e. for instance 
       going from back to max instead of min as the objective)"""
    obj_val, vals_of_vars = sol
    if problem.obj_sense == "max":
        obj_val *= -1.0
        
    vals_of_milp_vars = {var: int(val) if np.isclose(val, np.round(val), atol=1e-6) else val 
                         for var, val in zip(problem.vars, vals_of_vars)}

    return obj_val, vals_of_milp_vars
from typing import Tuple, Dict
from xion.models.milp import MILP
from xion.types import Vector, Scalar, Solution
from xion.core.presolver import RecoveryPipeline
import numpy as np

def convert_solver_result_to_MILP_result(sol: Solution, problem: MILP, rp: RecoveryPipeline) -> Tuple[float, Dict[str, Scalar]]:
    """Converts the solver results to the a human friendly format (i.e. for instance 
       going from back to max instead of min as the objective)"""
    # Load variable values both form the found solution (to the presolved problem and from the recovery pipeline.)
    obj_val, var_ass = sol
    vals_of_milp_vars = {var: (int(round(val)) if np.isclose(val, np.round(val), atol=1e-6) else val)
                         for var, val in zip(problem.vars, var_ass)}
    
    if problem.obj_sense == "max":
        obj_val *= -1.0

    return obj_val, vals_of_milp_vars
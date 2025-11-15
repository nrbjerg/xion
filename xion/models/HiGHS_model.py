import highspy as HiGHS
from xion.models.milp import MILP, Variable, Constraint, LinearCombination
import numpy as np 
from scipy.sparse import csc_matrix
from typing import Tuple, Dict
from xion.types import Vector, Scalar

def convert_MILP_to_HiGHS_lp_relaxation(problem: MILP) -> HiGHS.HighsLp:
    """Converts the passed MILP model to a HiGHS LP-relaxation which can be used directly within the xion MILP solver, by solving it with HiGHS."""
    lp = HiGHS.HighsLp()
    
    # 1. Add the variables to the LP-relaxation 
    lp.num_col_ = len(problem.vars)
    lp.num_row_ = len(problem.cons)
    if problem.obj_sense == "min":
        lp.col_cost_ = np.array([problem.obj_fun.weights.get(var, 0) for var in problem.vars], dtype=np.double)
    else:
        lp.col_cost_ = np.array([-problem.obj_fun.weights.get(var, 0) for var in problem.vars], dtype=np.double)
    lp.col_lower_ = np.array([var.l if not (var.l is None) else -HiGHS.kHighsInf for var in problem.vars], dtype=np.double)
    lp.col_upper_ = np.array([var.u if not (var.u is None) else HiGHS.kHighsInf for var in problem.vars], dtype=np.double)

    # 2. Add the constraints to the LP-relaxation.
    lp.row_lower_ = np.array([con.rhs if con.rel != "<=" else -HiGHS.kHighsInf for con in problem.cons], dtype=np.double)
    lp.row_upper_ = np.array([con.rhs if con.rel != ">=" else HiGHS.kHighsInf for con in problem.cons], dtype=np.double)
    # TODO: This is very inefficient - it is exactly what using sparse matrices avoids.
    A_dense = np.array([[con.lc.weights.get(var, 0.0) for var in problem.vars] for con in problem.cons], dtype=np.double)
    A_csc = csc_matrix(A_dense)
    lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_csc.data.astype(np.double)

    return lp

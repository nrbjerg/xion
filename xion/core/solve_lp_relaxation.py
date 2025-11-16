from highspy import Highs, HighsLp, HighsBasis, HighsSolution, HighsModelStatus
from typing import Optional, Tuple
from xion.types import Vector
import numpy as np

import highspy 
from xion.models.milp import MILP, Variable, Constraint, LinearCombination
import numpy as np 
from scipy.sparse import csc_matrix
from typing import Tuple, Dict
from xion.types import Vector, Scalar, Matrix

def generate_HiGHS_lp_relaxation(c: Vector, A: Matrix, b_leq: Vector, b_geq: Vector, lb: Vector, ub: Vector) -> highspy.HighsLp:
    """Converts the passed MILP model to a HiGHS LP-relaxation which can be used directly within the xion MILP solver, by solving it with HiGHS."""
    lp = highspy.HighsLp()
    
    # 1. Add the variables to the LP-relaxation 
    lp.num_col_ = A.shape[1]
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = c

    lp.col_lower_ = lb #np.array([var.lb if not (var.lb is None) else -HiGHS.kHighsInf for var in problem.vars], dtype=np.double)
    lp.col_upper_ = ub #np.array([var.ub if not (var.ub is None) else HiGHS.kHighsInf for var in problem.vars], dtype=np.double)

    # 2. Add the constraints to the LP-relaxation.
    lp.row_lower_ = b_leq # np.array([con.rhs if con.rel != "<=" else -HiGHS.kHighsInf for con in problem.cons], dtype=np.double)
    lp.row_upper_ = b_geq # np.array([con.rhs if con.rel != ">=" else HiGHS.kHighsInf for con in problem.cons], dtype=np.double)

    A_csc = csc_matrix(A)
    lp.a_matrix_.start_ = A_csc.indptr.astype(np.int32)
    lp.a_matrix_.index_ = A_csc.indices.astype(np.int32)
    lp.a_matrix_.value_ = A_csc.data.astype(np.double)

    return lp

def solve_lp_relaxation(lp_relaxation: highspy.HighsLp) -> Optional[Tuple[float, Vector]]:
    """Solves the provided LP-relaxation"""
    # Initialize HiGHS solver.
    h = Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("log_to_console", False)
    h.passModel(lp_relaxation)

    # Do the actual optimization 
    h.run()
    obj_val = h.getInfo().objective_function_value 
    sol = np.array(list(h.getSolution().col_value))
    print(sol)
    if h.getModelStatus() == HighsModelStatus.kOptimal and (not np.isnan(obj_val)):
        return (obj_val, sol)
    

#
#class LPRelaxationSolver:
#    """An LP-relaxation solver, used to solve the LP-relaxation problems throughout the branch and bound process."""
#
#    def __init__ (self): 
#        """Simply initialize the HiGHS backend."""
#
#    def solve (self, lp_relaxation: HighsLp, warm_start_basis: Optional[HighsBasis] = None) -> Optional[Tuple[float, HighsModelStatus, Vector, Optional[HighsBasis]]]:
#        """Solves the given lp_relaxation problem, via warm-starting if a basis is given."""
#        self.h.passModel(lp_relaxation)
#        self.h.setOptionValue("presolve", "on")
#        if not (warm_start_basis is None) and warm_start_basis.valid:
#            self.h.setBasis(warm_start_basis)
#
#        self.h.run()
#
#        # Check feasibility of the LP-relaxation
#        status = self.h.getModelStatus()
#        obj_val = self.h.getInfo().objective_function_value 
#        sol = np.array(list(self.h.getSolution().col_value))
#        if np.isnan(obj_val):
#            print("!", sol)
#        if status == HighsModelStatus.kOptimal and np.isnan(obj_val) == False:
#            basis = self.h.getBasis()
#            return (obj_val, status, sol, basis if basis.valid else None)

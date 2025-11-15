from highspy import Highs, HighsLp, HighsBasis, HighsSolution, HighsModelStatus
from typing import Optional, Tuple
from xion.types import Vector
import numpy as np

class LPRelaxationSolver:
    """An LP-relaxation solver, used to solve the LP-relaxation problems throughout the branch and bound process."""

    def __init__ (self): 
        """Simply initialize the HiGHS backend."""
        self.h = Highs()
        self.h.setOptionValue("output_flag", False)
        self.h.setOptionValue("log_to_console", False)
        self.h.setOptionValue("simplex_strategy", 1) # DUAL

    def solve (self, lp_relaxation: HighsLp, warm_start_basis: Optional[HighsBasis] = None) -> Optional[Tuple[float, HighsModelStatus, Vector, Optional[HighsBasis]]]:
        """Solves the given lp_relaxation problem, via warm-starting if a basis is given."""
        self.h.passModel(lp_relaxation)
        if not (warm_start_basis is None) and warm_start_basis.valid:
            self.h.setBasis(warm_start_basis)
        self.h.run()

        # Check feasibility of the LP-relaxation
        status = self.h.getModelStatus()
        obj_val = self.h.getInfo().objective_function_value 
        if status == HighsModelStatus.kInfeasible or np.isnan(obj_val):
            return None
        else:
            basis = self.h.getBasis()
            sol = np.array(list(self.h.getSolution().col_value))
            return [obj_val, status, sol, basis if basis.valid else None]

    
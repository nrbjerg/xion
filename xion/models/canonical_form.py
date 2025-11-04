from __future__ import annotations
from xion.models.milp import MILP, Variable, Constraint, LinearCombination
import numpy as np
from xion.types import Matrix, Vector, Scalar
from dataclasses import dataclass
from typing import Dict, List, Tuple
from copy import deepcopy

@dataclass
class CanonicalForm:
    """Models a MILP in canonical form i.e. min_x c^T x st. Ax = b and l <= x <= u"""
    # Coefficient vector, constraint matrix & vector
    c: Vector
    A: Matrix
    b: Vector
    
    # Information about the variables i.e. bounds & if they are integer valued
    l: Vector
    u: Vector
    integral_mask: Vector 
    integral_indices: Vector

    # TODO: Does not yet support unbound variables
    @staticmethod
    def from_milp(problem: MILP) -> CanonicalForm:
        """Converts the given MILP model to one in canonical form."""
        vars = deepcopy(problem.vars)

        # 1. Convert the constraints to equality constraints by adding slack variables
        cons: List[Constraint] = []
        for con in deepcopy(problem.cons):
            # NOTE: In this case we have nothing which needs to be done.
            if con.rel == "=": 
                cons.append(con)
                continue
            
            # NOTE: Add slack variable to constraint
            vars.append(Variable(f"@s{len(vars) - len(problem.vars)}", False, l = 0.0))
            if con.rel == "<=":
                con.lc += 1.0 * vars[-1]
            else:
                con.lc += -1.0 * vars[-1]

            con.rel = "="

            # NOTE: Make sure that the RHS is positive.
            if con.rhs < 0.0:
                con.lc *= -1.0
                con.rhs *= 1.0
            
            cons.append(con)

        # 2. Convert objective if needed
        if problem.obj_sense == "max":
            c = np.array([-problem.obj_fun.weights.get(var, 0.0) for var in vars])
        else:
            c = np.array([problem.obj_fun.weights.get(var, 0.0) for var in vars])

        # 3. Convert constraints to matrix & vector form.
        A = np.zeros((len(cons), len(vars)))
        b = np.zeros(len(cons))
        for j, con in enumerate(cons):
            for i, var in enumerate(vars):
                A[j, i] = con.lc.weights.get(var, 0.0)

            b[j] = con.rhs

        # 4. Generate lower and upper bounds for variables
        l = np.array([var.l if var.l is not None else -np.inf for var in vars])
        u = np.array([var.u if var.u is not None else np.inf for var in vars])

        # 5. Generate integral indices and mask
        integral_indices = np.array([i for i, var in enumerate(vars) if var.integral])
        integral_mask = np.array([True if var.integral else False for var in vars])

        return CanonicalForm(c, A, b, l, u, integral_mask, integral_indices)

    # TODO: Does not yet support unbound variables
    def convert_solution_to_milp_solution(self, sol: Tuple[float, Vector], problem: MILP) -> Tuple[float, Dict[Variable, Scalar]]:
        """Converts the solution to the problem in canonical form to one which is easily readable to MILP"""
        obj_val, vals_of_canonical_vars = sol
        if problem.obj_sense == "max":
            obj_val *= -1.0
        
        vals_of_milp_vars = {var: int(val) if np.isclose(val, np.round(val)) else val 
                            for var, val in zip(problem.vars, vals_of_canonical_vars)}

        return obj_val, vals_of_milp_vars

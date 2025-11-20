from __future__ import annotations
from xion.models.milp import MILP, Variable, Constraint, LinearCombination
import numpy as np
from xion.types import Matrix, Vector, Scalar, Solution
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from copy import deepcopy

@dataclass
class CanonicalForm:
    """Models a MILP in canonical form i.e. min_x c^T x st. A_leq x <= b_leq, A_eq x = b_eq and lb <= x <= ub"""
    # Coefficient vector, constraint matrices & vectors.
    c: Vector
    A_ineq: Matrix
    b_ineq: Vector
    A_eq: Matrix
    b_eq: Vector
    
    # Information about the variables i.e. bounds & if they are integer valued
    lb: Vector
    ub: Vector
    integral_mask: Vector 
    integral_indices: Vector

    cn: float = field(init=False, default=0.0)

    @staticmethod
    def from_milp(problem: MILP) -> CanonicalForm:
        """Converts the given MILP model to one in canonical form."""
        vars = deepcopy(problem.vars)
        n = len(vars)

        # 1. Convert the constraints to matrices & vectors
        eq_cons = deepcopy([con for con in problem.cons if con.rel == "="])
        ineq_cons = deepcopy([con for con in problem.cons if con.rel != "="])
        m_eq, m_ineq = len(eq_cons), len(ineq_cons)
        A_eq = np.empty((m_eq, n))
        b_eq = np.empty(m_eq)
        for i, con in enumerate(eq_cons):
            A_eq[i] = [con.lc.weights.get(var, 0.0) for var in vars]
            b_eq[i] = con.rhs
            
        A_ineq = np.empty((m_ineq, n))
        b_ineq = np.empty((m_ineq))
        for i, con in enumerate(ineq_cons):
            # If this is the case convert the >= constraint to a <= constraint
            if con.rel == ">=":
                con.lc *= -1.0
                con.rhs *= -1.0

            A_ineq[i] = [con.lc.weights.get(var, 0.0) for var in vars]
            b_ineq[i] = con.rhs
            
        # 2. Convert objective if needed
        c = np.array([problem.obj_fun.weights.get(var, 0.0) for var in vars])
        if problem.obj_sense == "max":
            c *= -1.0

        # 4. Generate lower and upper bounds for variables
        lb = np.array([var.lb if var.lb is not None else -np.inf for var in vars])
        ub = np.array([var.ub if var.ub is not None else np.inf for var in vars])

        # 5. Generate integral indices and mask
        integral_indices = np.array([i for i, var in enumerate(vars) if var.integral])
        integral_mask = np.array([True if var.integral else False for var in vars])

        return CanonicalForm(c, A_ineq, b_ineq, A_eq, b_eq, lb, ub, integral_mask, integral_indices)

    # TODO: Does not yet support unbound variables

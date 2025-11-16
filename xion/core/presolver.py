from xion.models.milp import MILP, Variable, Constraint
from dataclasses import dataclass, field
from xion.types import Scalar
from typing import Dict, Tuple, List
import numpy as np

@dataclass
class RecoveryPipeline:
    """A class which defines how variables can be recovered after preprocessing"""
    fixed_var_values: Dict[Variable, Scalar] = field(default_factory=dict)
    substitutions: Dict[Variable, Scalar] = field(default_factory=dict)

def tighten_variable_bounds(vars: List[Variable], cons: List[Constraint]) -> Tuple[List[Variable], bool]:
    """Tries to tighten the variable bounds, utilizing the constraints."""
    has_changed = False
    for var in vars:
        for con in cons:
            # We can only utilize constraints where the variable is present.
            if (var_coef := con.lc.weights.get(var, 0.0) != 0.0):
                if var_coef > 0:
                    weights = [con.lc.weights.get(v, 0.0) for v in vars if v != var]
                    # Proposition 7.5 Section 7.6
                    alternative_ub = (con.rhs - sum(w * (v.ub if w < 0 else v.lb) for w, v in zip(weights, vars) if v != var)) / var_coef
                    if alternative_ub < var.ub:
                        var.ub = alternative_ub
                        has_changed = True
                elif var_coef < 0:
                    weights = [con.lc.weights.get(v, 0.0) for v in vars if v != var]
                    # Proposition 7.5 Section 7.6
                    alternative_lb = (con.rhs - sum(w * (v.ub if w < 0 else v.lb) for w, v in zip(weights, vars) if v != var)) / var_coef
                    if alternative_lb > var.lb:
                        var.ub = alternative_lb
                        has_changed = True

    return vars, has_changed

def presolve(problem: MILP) -> Tuple[MILP, RecoveryPipeline]:
    """Preprocesses the MILP by strengthening bounds and removing redundant constraints"""
    non_fixed_vars = problem.vars
    # NOTE: for simplicity convert the cons so that they are either of type = or <=.
    cons = [con if con.rel != ">=" else Constraint((-1.0) * con.lc, "<=", -con.rhs) for con in problem.cons]
    rp = RecoveryPipeline()

    has_changed = True
    while has_changed == True:
        # 1. Perform bounds tightening to strengthen bounds on variables based on the constraints.
        non_fixed_vars, has_changed = tighten_variable_bounds(non_fixed_vars, cons)
        
        # 2. Do variable fixing to reduce the number of variables - and hence reduce the problem size.
        for var in non_fixed_vars:
            fixed_vars = dict() 

            # 2.1 If the variable is already fixed by its bounds, then fix it completely.
            if var.lb == var.ub:
                fixed_vars[var] = var.lb
                non_fixed_vars.remove(var)

            # 2.2 If the variable does no occur in any constraints, simply set it to the value which maximizes/minimizes the objective 
            # (alternatively if it is weighted 0, simply set it to 0 - this won't matter since it does not occur in any constraints).
            elif len(list(filter(lambda con: con.lc.weights.get(var, 0.0) != 0.0, cons))) == 0:
                coef_in_obj = problem.obj_fun.weights.get(var, 0.0)
                if coef_in_obj == 0.0: fixed_vars[var] = 0.0 
                elif coef_in_obj < 0.0: fixed_vars[var] = var.ub if problem.obj_sense == "min" else var.lb
                else: fixed_vars[var] = var.lb if problem.obj_sense == "max" else var.ub
                non_fixed_vars.remove()
            
            # 2.4 Utilize the integrality constraints
            if var.integral:
                var.ub = np.floor(var.ub)
                var.lb = np.ceil(var.lb)
            

        # Update the constraints to remove the fixed vars from them, NOTE: we don't need to do the 
        # same with the objective function since it is simply a transposition - and hence does not change
        # where the overall optima is located ;). However if the constraints are not changed, then 
        # the variable needs to be considered throughout the optimization process. But we technically also
        # do not need to remove them from the linear combination of the constraint, but we do anyway for simplicity.
        if len(fixed_vars) != 0:
            for i, con in enumerate(cons):
                cons[i].rhs -= sum(con.lc.weights.get(fixed_var, 0.0) * val for fixed_var, val in fixed_vars.items())
                cons[i].lc.weights = {var: weight for var, weight in con.lc.weights.items() if not (var in fixed_vars.keys())}

            rp.fixed_var_values.update(fixed_vars)
            has_changed = True
            
        # 3. Remove redundant constraints
        

        # 4. Row / column singletons, these can similarly be used to fix variables etc.

        
        
    # Construct the simplified 
    presolved_milp = MILP(f"{problem.identifier}[presolved]", non_fixed_vars, cons, problem.obj_fun, problem.obj_sense)
    return (presolved_milp, rp)

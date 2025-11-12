import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_SCP(n: int, m: int, seed: int, density: float = 0.1) -> Tuple[float, MILP]:
    """Generates and solves a Set Covering Problem, returning both the objective value and the MILP model"""
    np.random.seed(seed)
    covers = (np.random.uniform(size=(m, n)) < density).astype(int)
    for i in range(n):
        if covers[:, i].sum() == 0: # NOTE: ensure that each element i are covered by at least one set, so that the problem is not infeasible.
            covers[np.random.randint(0, m), i] = 1
    
    costs = np.random.uniform(size=m)

    # Compute optimal value using Gurobi
    model = gp.Model(f"SCP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(costs[j] * x[j] for j in range(m)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum(covers[j, i] * x[j] for j in range(m)) >= 1.0 for i in range(n)))
    model.optimize()

    # Setup MILP 
    xs = [Variable.new_binary(f"x{j}") for j in range(m)]
    obj_fun = sum([costs[j] * xs[j] for j in range(m)])
    cons = [Constraint(sum(covers[j, i] * xs[j] for j in range(m)), ">=", 1.0) for i in range(n)]

    return (model.ObjVal, MILP(f"SCP{seed}", xs, cons, obj_fun, obj_sense="min"))
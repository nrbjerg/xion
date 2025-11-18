import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_SCP(n: int, m: int, seed: int, density: float = 0.1) -> Tuple[float, MILP]:
    """Generates and solves a Set Covering Problem (n: sets, m: items), returning both the objective value and the MILP model"""
    np.random.seed(seed)
    covers = (np.random.uniform(size=(n, m)) < density).astype(int)
    for j in range(m):
        if covers[:, j].sum() == 0: # NOTE: ensure that each element i are covered by at least one set, so that the problem is not infeasible.
            covers[np.random.randint(0, n), j] = 1
    
    costs = np.random.uniform(size=n)

    # Compute optimal value using Gurobi
    model = gp.Model(f"SCP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(costs[i] * x[i] for i in range(n)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum(covers[i, j] * x[i] for i in range(n)) >= 1.0 for j in range(m)))
    model.optimize()

    # Setup MILP 
    xs = [Variable.new_binary(f"x{i}") for i in range(n)]
    obj_fun = sum([costs[i] * xs[i] for i in range(n)])
    cons = [Constraint(sum(covers[i, j] * xs[i] for i in range(n)), ">=", 1.0) for j in range(m)]

    np.random.shuffle(xs)
    return ((model.ObjVal, model.Runtime), MILP(f"SCP{seed}", xs, cons, obj_fun, obj_sense="min"))
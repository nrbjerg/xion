import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple, Optional
from xion.models.milp import MILP, Constraint, Variable

def generate_BP(n: int, m: int, seed: int) -> Tuple[Optional[float], MILP]:
    """Generates and solves a 1D Bin-Packing Problem (n: items, m: maximum number of bins), returning both the objective value and the MILP model"""
    np.random.seed(seed)
    sizes = np.round(np.random.uniform(size=n), 3)
    C = max(max(sizes), sum(sizes) / m) * (1 + np.abs(np.random.normal())) 

    # Compute exact solution using Gurobi
    model = gp.Model(f"BP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, m, vtype=GRB.BINARY, name="x")
    y = model.addVars(m, vtype=GRB.BINARY, name="y")

    model.setObjective(gp.quicksum(y[j] for j in range(m)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum(sizes[i] * x[i,j] for i in range(n)) - C * y[j] <= 0 for j in range(m)))
    model.addConstrs((gp.quicksum(x[i,j] for j in range(m)) == 1.0 for i in range(n)))
    model.optimize()

    # Setup MILP
    ys = [Variable.new_binary(f"y{i}") for i in range(m)]
    xs = [[Variable.new_binary(f"x{i, j}") for j in range(m)] for i in range(n)]
    obj_fun = sum(ys[j] for j in range(m))
    cons = ([Constraint(sum(sizes[i] * xs[i][j] for i in range(n)) - C * ys[j], "<=", 0) for j in range(m)] + 
            [Constraint(sum(xs[i][j] for j in range(m)), "=", 1.0) for i in range(n)])

    return ((model.ObjVal, model.Runtime), MILP(f"BP{seed}", ys + sum(xs, []), cons, obj_fun, obj_sense="min"))


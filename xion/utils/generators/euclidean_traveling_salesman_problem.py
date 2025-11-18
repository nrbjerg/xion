import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_ETSP(n: int, seed: int) -> Tuple[float, MILP]:
    """Generates and solves an euclidian Traveling Salesman Problem with MTZ subtour elimination constraints,
       returning both the objective value and the MILP model"""
    np.random.seed(seed)
    positions = np.random.uniform(size=(n, 2)) 
    dists = np.array([[np.linalg.norm(p - q) for p in positions] for q in positions])

    # Compute optimal value using Gurobi
    model = gp.Model(f"ETSP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, n, vtype=GRB.BINARY)
    u = model.addVars(n, vtype=GRB.CONTINUOUS, lb = 1.0, ub = float(n))
    model.setObjective(gp.quicksum(dists[i, j] * x[i, j] for i in range(n) for j in range(n)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum(x[i,j] for j in range(n) if i != j) == 1.0 for i in range(n)))
    model.addConstrs((gp.quicksum(x[j,i] for j in range(n) if i != j) == 1.0 for i in range(n)))
    model.addConstr(u[0] == 1)
    model.addConstrs((u[i] - u[j] + n * x[i, j] <= n - 1) for i in range(1, n) for j in range(1, n) if i != j)
    model.optimize()

    # Setup MILP 
    xs = [[Variable.new_binary(f"x{i, j}") for i in range(n)] for j in range(n)]
    us = [Variable.new_continuous(f"u{i}", lb = 1.0, ub = float(n)) for i in range(n)]
    obj_fun = sum(dists[i, j] * xs[i][j] for i in range(n) for j in range(n))
    cons = ([Constraint(sum(xs[i][j] for j in range(n) if i != j), "=", 1.0) for i in range(n)] + 
            [Constraint(sum(xs[j][i] for j in range(n) if i != j), "=", 1.0) for i in range(n)] + 
            [Constraint(us[0], "=", 1)] +
            [Constraint(us[i] - us[j] + n * xs[i][j], "<=", n - 1) for i in range(1, n) for j in range(1, n) if i != j])

    vars = us + sum(xs, [])
    np.random.shuffle(vars)
    return ((model.ObjVal, model.Runtime), MILP(f"ETSP{seed}", vars, cons, obj_fun, obj_sense="min"))
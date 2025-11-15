import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple, Optional
from xion.models.milp import MILP, Constraint, Variable

def generate_IGAP(n: int, m: int, seed: int) -> Tuple[Optional[float], MILP]:
    """Generates and solves a Integer Generalized Assignment Problem (n: jobs, m: agents), returning both the objective value and the MILP model"""
    np.random.seed(seed)
    resource_utilization = np.random.uniform(size=(n,m))
    capacities = np.random.lognormal(size=m) * max(np.sum(resource_utilization, axis=1))
    max_units = np.random.randint(1, 5, size=(n)) 
    profits = 2 * resource_utilization + np.random.uniform(size=(n,m))

    # Compute exact solution using Gurobi
    model = gp.Model(f"IGAP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, m, vtype=GRB.INTEGER, name="x", lb=0, ub={(i, j): max_units[i] for i in range(n) for j in range(m)})

    model.setObjective(gp.quicksum(profits[i, j] * x[i, j] for i in range(n) for j in range(m)), GRB.MAXIMIZE)
    model.addConstrs((gp.quicksum(resource_utilization[i, j] * x[i,j] for i in range(n)) <= capacities[j] for j in range(m)))
    model.addConstrs((gp.quicksum(x[i,j] for j in range(m)) <= max_units[i] for i in range(n)))
    model.optimize()

    # Setup MILP
    xs = [[Variable.new_integer(f"x{i, j}", l = 0, u=max_units[i]) for j in range(m)] for i in range(n)]
    obj_fun = sum(profits[i, j] * xs[i][j] for i in range(n) for j in range(m))
    cons = ([Constraint(sum(resource_utilization[i, j] * xs[i][j] for i in range(n)), "<=", capacities[j]) for j in range(m)] + 
            [Constraint(sum(xs[i][j] for j in range(m)), "<=", max_units[i]) for i in range(n)])

    return (model.ObjVal, MILP(f"IGAP{seed}", sum(xs, []), cons, obj_fun, obj_sense="max"))


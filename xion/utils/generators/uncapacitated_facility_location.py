import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_UFL(n: int, m: int, seed: int) -> Tuple[float, MILP]:
    """Generates and solves a Uncapacitated Facility Location Problem (n: customers, m: facility locations), returning both the objective value and the MILP model"""
    np.random.seed(seed)
    opening_costs = np.random.lognormal(size=m) 
    costs = np.random.lognormal(size=(n, m))

    # Compute optimal value using Gurobi
    model = gp.Model(f"CFLP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, m, vtype=GRB.BINARY)
    y = model.addVars(m, vtype=GRB.BINARY)

    model.setObjective(gp.quicksum(opening_costs[j] * y[j] for j in range(m)) + gp.quicksum(costs[i, j] * x[i, j] for j in range(m) for i in range(n)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum(x[i,j] for j in range(m)) == 1.0 for i in range(n)))
    model.addConstrs((x[i,j] <= y[j] for j in range(m) for i in range(n)))
    model.optimize()

    # Setup MILP 
    ys = [Variable.new_binary(f"y{j}") for j in range(m)]
    xs = [[Variable.new_binary(f"x{i, j}") for j in range(m)] for i in range(n)]
    obj_fun = (sum([opening_costs[j] * ys[j] for j in range(m)]) + 
               sum(sum(costs[i, j] * xs[i][j] for i in range(n)) for j in range(m)))
    cons = ([Constraint(sum(xs[i][j] for j in range(m)), "=", 1.0) for i in range(n)] + 
            [Constraint(xs[i][j] - ys[j], "<=", 0.0) for j in range(m) for i in range(n)]) 

    return (model.ObjVal, MILP(f"SCP{seed}", ys + sum(xs, []), cons, obj_fun, obj_sense="min"))
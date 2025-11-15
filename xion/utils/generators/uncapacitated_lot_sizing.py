import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_ULS(n: int, seed: int) -> Tuple[float, MILP]:
    """Generates and solves an Uncapacitated Lot-Sizing Problem (n: time periods), returning both the objective value and the MILP model"""
    np.random.seed(seed)
    demands = np.random.lognormal(mean=2.0, sigma=0.5, size=n)
    setup_costs = np.random.uniform(0, 50, size=n)
    production_costs = np.random.uniform(0, 10, size=n)
    holding_costs = np.random.uniform(0, 5, size=n)
    M = sum(demands)

    # Objective: Minimize total cost
    model = gp.Model(f"ULS{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="x")  # production
    i = model.addVars(n, vtype=GRB.CONTINUOUS, lb=0.0, name="i")  # inventory
    y = model.addVars(n, vtype=GRB.BINARY, name="y")               # setup

    model.setObjective(
        gp.quicksum(setup_costs[t] * y[t] + production_costs[t] * x[t] + holding_costs[t] * i[t] for t in range(n)),
        GRB.MINIMIZE
    )
    # Compute optimal value using Gurobi

    model.setObjective(gp.quicksum(setup_costs[t] * y[t] + production_costs[t] * x[t] + holding_costs[t] * i[t] for t in range(n)), GRB.MINIMIZE)
    model.addConstr(i[0] == 0.0)
    model.addConstr(x[0] >= demands[0])
    model.addConstrs((i[t-1] + x[t] - i[t] >= demands[t] for t in range(1, n)))
    model.addConstrs((x[t] - M * y[t] <= 0.0 for t in range(n)))
    model.optimize()

    # Setup MILP 
    xs = [Variable.new_continuous(f"x{t}", l=0.0) for t in range(n)]
    ys = [Variable.new_binary(f"y{t}") for t in range(n)]
    i_vars = [Variable.new_continuous(f"i{t}", l=0.0) for t in range(n)]
    obj_fun = sum(setup_costs[t] * ys[t] + production_costs[t] * xs[t] + holding_costs[t] * i_vars[t] for t in range(n))
    cons = ([Constraint(i_vars[0], "=", 0.0)] +
            [Constraint(xs[0], ">=", demands[0])] +
            [Constraint(i_vars[t-1] + xs[t] - i_vars[t], ">=", demands[t]) for t in range(1, n)] + 
            [Constraint(xs[t] - M * ys[t], "<=", 0.0) for t in range(n)])

    return ((model.ObjVal, model.Runtime), MILP(f"ULS{seed}", ys + xs + i_vars, cons, obj_fun, obj_sense="min"))
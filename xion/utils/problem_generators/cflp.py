import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_CFLP(n: int, m: int, seed: int) -> Tuple[float, MILP]:
    """Generates and solves a Set Covering Problem, returning both the objective value and the MILP model"""
    np.random.seed(seed)
    opening_cost = np.random.uniform(10, 100, size=m) # Facility opening cost
    capacities = np.random.uniform(n / m, 4 * n / m, size=m) 
    strain = np.random.uniform(size=(m, n))
    costs = np.random.uniform(size=(m, n))

    # Compute optimal value using Gurobi
    model = gp.Model(f"CFLP{seed}")
    model.setParam("OutputFlag", 0)

    x = model.addVars(m, n, name="x", lb = 0.0, ub = 1.0)
    y = model.addVars(m, vtype=GRB.BINARY)

    model.setObjective(gp.quicksum(opening_cost[j] * y[j] for j in range(m)) + gp.quicksum(costs[j, i] * x[j, i] for j in range(m) for i in range(n)), GRB.MINIMIZE)
    model.addConstrs((gp.quicksum(x[j,i] * strain[j, i] for i in range(n)) <= capacities[j] * y[j] for j in range(m)))
    model.addConstrs((gp.quicksum(x[j,i] for j in range(m)) == 1.0 for i in range(n)))
    model.optimize()

    # Setup MILP 
    ys = [Variable.new_binary(f"y{j}") for j in range(m)]
    xs = [[Variable.new_continuous(f"x{j, i}", l = 0.0, u = 1.0) for i in range(n)] for j in range(m)]

    obj_fun = sum([opening_cost[j] * ys[j] for j in range(m)]) + sum(sum(costs[j, i] * xs[j][i] for i in range(n)) for j in range(m))
    cons = ([Constraint(sum(xs[j][i] for j in range(m)), "=", 1.0) for i in range(n)] + 
            [Constraint(sum(strain[j, i] * xs[j][i] for i in range(n)) + (-capacities[j] * ys[j]), "<=", 0.0) for j in range(m)]) 

    return (model.ObjVal, MILP(f"SCP{seed}", ys + sum(xs, []), cons, obj_fun, obj_sense="min"))
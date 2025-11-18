import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_BMDKP(n: int, m: int, seed: int) -> Tuple[float, MILP]:
    """Generates and solves a Binary Multi Dimensional Knapsack Problem (n: items, m: dimensions), returning both the objective value and the MILP model"""
    np.random.seed(seed)
    values = np.round(np.random.uniform(size=n), 3)
    weights = np.round(np.random.uniform(size=(m, n)), 3)
    capacities = np.round((0.3 * weights.sum(axis=1)), 3)

    # Compute exact solution using Gurobi
    model = gp.Model(f"BMDKP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(values[i] * x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstrs((gp.quicksum(weights[j, i] * x[i] for i in range(n)) <= capacities[j] for j in range(m)))
    model.optimize()

    # Setup MILP
    xs = [Variable.new_binary(f"x{i}") for i in range(n)]
    cons = [Constraint(sum(weights[j, i] * xs[i] for i in range(n)), "<=", capacities[j]) for j in range(m)]
    obj_fun = sum(values[i] * xs[i] for i in range(n))

    np.random.shuffle(xs)
    return ((model.ObjVal, model.Runtime), MILP(f"BMDKP{seed}", xs, cons, obj_fun, obj_sense="max"))


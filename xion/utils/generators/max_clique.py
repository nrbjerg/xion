import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Tuple
from xion.models.milp import MILP, Constraint, Variable

def generate_MC(n: int, seed: int, density: float = 0.8) -> Tuple[float, MILP]:
    """Generates and solves a Max-Clique Problem, returning both the objective value and the MILP model"""
    np.random.seed(seed)
    adjacency_matrix = np.random.choice(2, size=(n, n), replace=True, p=[1 - density, density])

    # Compute optimal value using Gurobi
    model = gp.Model(f"SCP{seed}")
    model.setParam("OutputFlag", 0)
    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    model.addConstrs((x[i] + x[j] <= 1.0 for j in range(n) for i in range(n) if i != j and adjacency_matrix[i,j] != 1))
    model.optimize()

    # Setup MILP 
    xs = [Variable.new_binary(f"x{j}") for j in range(n)]
    obj_fun = sum([xs[j] for j in range(n)])
    cons = [Constraint(xs[i] + xs[j], "<=", 1.0) for j in range(n) for i in range(n) if i != j and adjacency_matrix[i, j] != 1]

    return (model.ObjVal, MILP(f"MC{seed}", xs, cons, obj_fun, obj_sense="max"))
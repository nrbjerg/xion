from xion import solve, MILP, Variable, Constraint
import xion
import numpy as np
from typing import Tuple, List, Dict
import time
import json
import os 
from tqdm import tqdm

import gurobipy as gp
from gurobipy import GRB

# Runs 3 benchmark problems, namely small sized TSP (20 nodes), a knapsack problem () & 
def generate_BMDKP(n: int, m: int, seed: int) -> Tuple[float, MILP]:
    """Generates and solves a Binary Multi Dimensional Knapsack Problem, returning both the objective value and the MILP model"""
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

    return (model.ObjVal, MILP(f"BMDKP{seed}", xs, cons, obj_fun, obj_sense="max"))

def generate_TSP(n: int, seed: int) -> Tuple[float, MILP]:
    pass 

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

def benchmark(repeats: int = 8) -> float:
    problem_types_and_generators = {
        #"BMDKP": lambda seed: generate_BMDKP(64, 16, seed = seed),
        #"SCP": lambda seed: generate_SCP(1024, 96, seed = seed),
        "CFLP": lambda seed: generate_CFLP(256, 64, seed = seed),
    }
    times: Dict[str, List[float]] = {}
    for problem_type, generator in problem_types_and_generators.items():
        print(f"Currently benchmarking on {problem_type}")
        times[problem_type] = []
        for seed in tqdm(range(repeats)):
            obj_val_from_gurobi, milp = generator(seed)
            start_time = time.time()
            obj_val_from_xion, _ = solve(milp)
            times[problem_type].append(time.time() - start_time)
            assert np.isclose(obj_val_from_gurobi, obj_val_from_xion)

    # Log the run times 
    with open(os.path.join(os.getcwd(), "benchmarks", f"xion{xion.__version__}.json"), "w+") as file:
        json.dump(times, file)

if __name__ == "__main__":
    benchmark(1)
    
    
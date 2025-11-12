from xion import solve, MILP, Variable, Constraint
import xion
import numpy as np
from typing import Tuple, List, Dict
import time
import json
import os 
from tqdm import tqdm
from xion.utils.problem_generators.bmdkp import generate_BMDKP
from xion.utils.problem_generators.scp import generate_SCP
from xion.utils.problem_generators.cflp import generate_CFLP

def benchmark(repeats: int = 8) -> float:
    """Runs a benchmark on some simple MILP problems."""
    problem_types_and_generators = {
        "BMDKP": lambda seed: generate_BMDKP(64, 32, seed = seed),
        "SCP": lambda seed: generate_SCP(512, 96, seed = seed),
        "CFLP": lambda seed: generate_CFLP(128, 48, seed = seed),
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
    benchmark(repeats=16)
    #obj_val_from_gurobi, milp = generate_BMDKP(64, 32, seed = 9)
    #obj_val_from_xion, _ = solve(milp)
    #print(obj_val_from_gurobi, obj_val_from_xion)
    
    
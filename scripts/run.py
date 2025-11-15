from xion import solve, MILP, Variable, Constraint
import xion
import numpy as np
from typing import Tuple, List, Dict, Callable
import time
import json
import os 
from tqdm import tqdm
from xion.types import Scalar
from xion.utils.generators.integer_generalized_assignment_problem import generate_IGAP # NOTE: To test performance on integer variables.
from xion.utils.generators.bin_packing import generate_BP # NOTE: Combinatorial assignment, tests strong branching effectiveness.
from xion.utils.generators.set_covering import generate_SCP # NOTE: Huge number of variables & sparse constraint matrix.
from xion.utils.generators.binary_multi_dimensional_knapsack_problem import generate_BMDKP # NOTE: LP root quality vs integrality gap, primal heuristics.
from xion.utils.generators.max_clique import generate_MC # NOTE: Tests branching decisions & tiny integrality gaps.
from xion.utils.generators.uncapacitated_facility_location import generate_UFL # NOTE: Tests warm starts, big-M usage.
from xion.utils.generators.uncapacitated_lot_sizing import generate_ULS # NOTE: Tests continuous variables and big-M usage.
from xion.utils.generators.traveling_salesman_problem import generate_TSP # NOTE: Just for fun :)
from loguru import logger

from xion.utils.evaluator import evaluate_milp_at_var_ass
from xion.utils.feasibility import compute_number_of_violated_constraints

def benchmark(repeats: int = 8) -> float:
    """Runs a benchmark on some simple MILP problems."""
    problem_types_and_generators: Dict[str, Callable[[int], Tuple[Scalar, MILP]]] = {
        "IGAP(n=32, m=12)": lambda seed: generate_IGAP(32, 12, seed=seed),
        "SCP(n=1024, m=256)": lambda seed: generate_SCP(1024, 256, density = 0.02, seed = seed),
        "UFL(n=512, m=48)": lambda seed: generate_UFL(512, 48, seed = seed),
        "MC(n=128)": lambda seed: generate_MC(128, density=3/4, seed=seed),
        "TSP(n=24)": lambda seed: generate_TSP(16, seed=seed),
        "BP(n=512, m=32)": lambda seed: generate_BP(512, 32, seed=seed),
        #"ULS(n=24)": lambda seed: generate_ULS(24, seed=seed),
    } 
    times: Dict[str, List[float]] = {}
    for problem_type, generator in problem_types_and_generators.items():
        print(f"Currently benchmarking on {problem_type}")
        times[problem_type] = []
        for seed in tqdm(range(repeats)):
            obj_val_from_gurobi, milp = generator(seed)
            start_time = time.perf_counter()
            obj_val_from_xion, var_ass_from_xion = solve(milp)
            times[problem_type].append(time.perf_counter() - start_time)
            if not np.isclose(obj_val_from_gurobi, obj_val_from_xion):
                if (compute_number_of_violated_constraints(milp, var_ass_from_xion) == 0 and ((obj_val_from_xion > obj_val_from_gurobi and milp.obj_sense == "max") or 
                                                                                                (obj_val_from_xion < obj_val_from_gurobi and milp.obj_sense == "min"))):
                    logger.info(f"Found a better solution than gurobi on {problem_type}{seed}")
                else:
                    logger.error(f"There is a bug in the solver (found on {problem_type}{seed})")
            

    # Log the run times 
    with open(os.path.join(os.getcwd(), "benchmarks", f"xion{xion.__version__}.json"), "w+") as file:
        json.dump(times, file)

if __name__ == "__main__":
    (obj_val_from_gurobi, gurobi_run_time), milp = generate_BMDKP(256, 12, seed = 0)
    obj_val_from_xion, var_ass_from_xion = solve(milp, verbose=True)
    print(f"xion obj: {evaluate_milp_at_var_ass(milp, var_ass_from_xion):.3f}, number of cons violations: {compute_number_of_violated_constraints(milp, var_ass_from_xion)}")
    print(f"gurobi obj: {obj_val_from_gurobi:.3f}, gurobi run time: {gurobi_run_time:.1f}")
    #print(obj_val_from_gurobi, obj_val_from_xion, time.perf_counter() - start_time)

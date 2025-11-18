#from xion import solve
from xion.core.solver import solve
from xion.types import Scalar
from xion.utils.generators.integer_generalized_assignment_problem import generate_IGAP # NOTE: To test performance on integer variables.
from xion.utils.generators.bin_packing import generate_BP # NOTE: Combinatorial assignment, tests strong branching effectiveness.
from xion.utils.generators.set_covering import generate_SCP # NOTE: Huge number of variables & sparse constraint matrix.
from xion.utils.generators.binary_multi_dimensional_knapsack_problem import generate_BMDKP # NOTE: LP root quality vs integrality gap, primal heuristics.
from xion.utils.generators.max_clique import generate_MC # NOTE: Tests branching decisions & tiny integrality gaps.
from xion.utils.generators.uncapacitated_facility_location import generate_UFL # NOTE: Tests warm starts, big-M usage.
from xion.utils.generators.uncapacitated_lot_sizing import generate_ULS # NOTE: Tests continuous variables and big-M usage.
from xion.utils.generators.euclidean_traveling_salesman_problem import generate_ETSP # NOTE: Just for fun :)
from loguru import logger

from xion.utils.evaluator import evaluate_milp_at_var_ass
from xion.utils.feasibility import compute_number_of_violated_constraints

if __name__ == "__main__":

    (obj_val_from_gurobi, gurobi_run_time), milp = generate_BMDKP(128, 8, seed=6)
    logger.info(f"gurobi obj: {obj_val_from_gurobi:.3f}, gurobi run time: {gurobi_run_time:.3f}s")
    obj_val_from_xion, var_ass_from_xion = solve(milp, verbose=True)
    logger.info(f"xion obj: {evaluate_milp_at_var_ass(milp, var_ass_from_xion):.3f}, number of cons violations: {compute_number_of_violated_constraints(milp, var_ass_from_xion, verbose=True)}")
    logger.info(f"xion var_ass: {var_ass_from_xion}")
    #print(obj_val_from_gurobi, obj_val_from_xion, time.perf_counter() - start_time)

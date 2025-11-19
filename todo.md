# V0.0.1 Basic functionality
- [x] Implement a basic way to formulate MILP problems
- [x] Implement basic branch-and-bound solver
# V0.0.2 Branch-and-Bound improvements
- [x] Implement best-bound node selection
# V0.0.3 Direct HiGHS interface & warm starting
- [x] Move directly to HiGHS instead of working through the scipy wrapper.
- [x] Implement warm starting HiGHS
# V0.0.4 Improved Benchmarking.
- [x] Implement a time budget for the solver.
- [x] Add benchmarking visualization
- [-] Add extra MILP problems:
   - [x] 1D Bin Packing
   - [x] Integer Generalized Assignment Problem
   - [x] Max Clique
   - [x] Traveling Salesman (MTZ formulation)
   - [x] Uncapacitated Facility Location
   - [x] Uncapacitated lot sizing
- [x] Create a variable assignment evaluator.
- [x] Create a feasibility checker.
# V0.0.5 Reliability Branching
- [x] Reliability Branching
   - [x] Implement strong-branching
   - [x] Implement pseudo-costs 
- [ ] Check if optimal solution must be integer.
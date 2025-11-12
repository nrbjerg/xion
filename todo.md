# V0.0.1 Basic functionality
- [x] Implement a basic way to formulate MILP problems
- [x] Implement basic branch-and-bound solver
# V0.0.2 Branch-and-Bound improvements
- [x] Implement best-bound node selection
- [x] Implement strong branching
# V0.0.3 Direct HiGHS interface & warm starting
- [x] Move directly to HiGHS instead of working through the scipy wrapper.
- [x] Implement warm starting HiGHS
# V0.0.4 Quality of life & improved benchmarking
- [-] Implement a time budget for the solver.
- [ ] Improve python MILP interface 
   - [ ] Add support for variables on the RHS when defining constraints 
   - [ ] Add support for the use of <= = and >= directly within python to create constraints
   - [ ] Implement variables which may be unbounded (i.e. -inf).
- [-] Add benchmarking visualization
- [ ] Implement callback for lazily adding constraints - for instance adding DFJ (subtour elimination) constraints to TSP.

# Xion
A *pure* python implementation of a branch and bound MILP solver, originally created for educational purposes. This has however changed as development progressed - The behind the development of Xion is now to to make the *fastest pure python MILP solver* possible. 

I say *pure* since it uses HiGHS (for now through scipy.optimize.linprog) to solve the LP-relaxations of the MILP under the hood, which is a major part of the computational burden when solving MILP problems. 

## Example: Solving the 0-1 Knapsack problem using Xion 
Below you will find a simple illustrative example of how the Xion library can be utilized to solve a simple binary knapsack problem.

```python
from xion import MILP, Constraint, Variable, solve
import numpy as np

N = 16 # Number of items  
w = np.random.uniform(size=N) # Weights - i.e. w[i] is the weight of item i.
v = np.random.uniform(size=N) # Values - i.e. v[i] is the weight of item i is the weight of item i.
C = sum(w) / 2 # Capacity of the knapsack.

# Define the problem, using the Xion package
xs = [Variable.new_binary(f"x{i}") for i in range(N)] 
cons = [Constraint(sum(w[i] * xs[i] for i in range(N)), "<=", C)]
obj_fun = sum(v[i] * xs[i] for i in range(N))
problem = MILP("knapsack problem", xs, cons, obj_fun, obj_sense="max")

# solve the actual MILP using Xion.
if (sol := solve(problem, verbose=True)) != None: # solve will return None if the problem is infeasible.
    obj_val, var_ass = sol
    print(f"Found an optimal solution to the knapsack problem, with an objective of {obj_val:.3f}, the solution is:")
    for x in xs:
        print(f"{x} = {var_ass[x]}")
```
For more examples please check the [xion/utils/generators](https://github.com/nrbjerg/xion/tree/main/xion/utils/generators) directory, which contains several general well known MILP problem formulations utilized for benchmarking the Xion solver.

## Install guide
Currently the Xion package is not available through PyPi package manager.

However to *install* Xion, from source, simply clone the repository, install the requirements through pip and install the package. This can be accomplished by running the following commands:
```
git clone https://github.com/nrbjerg/xion
cd xion
pip install -r requirements.txt
pip install .
```

To *uninstall* Xion along with any installed *all* of its requirements (**NOTE:** Please check to see if requirements.txt contains any previously installed packages and act accordingly.) simply run:
```
pip uninstall -r requirements.txt
pip uninstall xion
```

## Features
At the moment, the following features is implemented within the Xion solver:
- [x] *Reliability Branching* on integral variables.
- [x] Basic *Branch and Bound* search using *Best-Bound Node Selection*.
- [x] Basic *library interface*, as illustrated in the knapsack example.

### Development Roadmap
Below is a list of features which are currently in the works for the main solver:

**Heuristics:** 
- [ ] *Feasibility Pump* (objective version)
- [ ] *Relaxation Induced Neighborhood Search (RINS)*

**Presolving:** (TBD)

**Cutting Planes:** (TBD)

**General Solver Improvements:**
  - [ ] *Hybrid Branching* by combining: *Inference & Reliability Branching*

**Interface Improvements:** 
  - [ ] Support for *lazy constraint callbacks* (such as lazily adding DFJ subtour elimination constraints for the TSP).
  - [ ] Utilize the comparisons operators: '<=', '=' and '>=', directly to construct constraints through operator overloading.
  - [ ] Add support for variables on the right hand side of constraints.
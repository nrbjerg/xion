# Xion
A "pure" python implementation of a branch and bound MILP solver, created for educational purposes. I say "pure" since it uses HiGHS (through scipy.optimize.linprog) to solve the LP-relaxations of the MILP under the hood, which is a major part of the computational burden when solving MILP problems.

## Install
To install xion (locally) simply clone the repository, install the requirements through pip and install the package via the following commands:
```
git clone https://github.com/nrbjerg/xion
pip install -r requirements.txt
pip install .
```

## Key Features 


## Example
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

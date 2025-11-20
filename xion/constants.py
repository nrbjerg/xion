import highspy

THETA = 1e-3 # A lower bound for when a change is sufficient, for instance see the root presolving function.
EPSILON = 1e-6 # When is two floating point numbers sufficiently to be considered identical (also known as atol in numpy)
INF = highspy.kHighsInf
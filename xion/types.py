from numpy.typing import ArrayLike
from typing import Literal, Union

Matrix = ArrayLike
Vector = ArrayLike
Scalar = Union[float, int]

ComparisonOperator = Literal["=", ">=", "<="]
ObjectiveSense = Literal["min", "max"]


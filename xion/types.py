from numpy.typing import ArrayLike
from typing import Literal, Union, Tuple

Matrix = ArrayLike
Vector = ArrayLike
Scalar = Union[float, int]
Solution = Tuple[Scalar, Vector]

ComparisonOperator = Literal["=", ">=", "<="]
ObjectiveSense = Literal["min", "max"]

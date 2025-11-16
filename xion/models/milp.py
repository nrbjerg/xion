from __future__ import annotations
from dataclasses import dataclass, field
from xion.types import Matrix, Vector, Scalar, ComparisonOperator, ObjectiveSense
from typing import List, Optional, Union, Dict
import numpy as np

@dataclass
class Variable:
    """Models a variable in a MILP."""
    identifier: str
    integral: bool
    lb: Optional[float] = None
    ub: Optional[float] = None

    @staticmethod
    def new_binary(identifier: str) -> Variable:
        """Initializes a new binary variable"""
        return Variable(identifier, True, lb = 0.0, ub = 1.0)

    @staticmethod
    def new_integer(identifier: str, lb: Optional[float] = None, ub: Optional[float] = None) -> Variable:
        """Initializes a new integer variable"""
        return Variable(identifier, True, lb = lb, ub = ub)

    @staticmethod
    def new_continuous(identifier: str, lb: Optional[float] = None, ub: Optional[float] = None) -> Variable:
        """Initializes a new continuous variable"""
        return Variable(identifier, False, lb = lb, ub = ub)

    def __add__(self, other: Union[Variable, LinearCombination]) -> LinearCombination:
        """Adds two variables together or adds the variable to a linear combination."""
        if isinstance(other, (int, float)) and (other == 0.0 or other == 0): # NOTE: needed to enable the use of the sum function.
            return self

        if not isinstance(other, (Variable, LinearCombination)):
            raise TypeError(f"Variable {self.identifier} was summed with something which was not another Variable or a LinearCombination: {other} of type {type(other)}")
            
        if isinstance(other, Variable):
            return LinearCombination({self: 1.0, other: 1.0})

        else:
            # Add the variable to the other linear combination
            other.weights[self] = 1.0
            return other 

    def __radd__ (self, other) -> LinearCombination:
        """Needed to enable addition."""
        return self.__add__(other)

    # FIXME: This might introduce a bug
    def __sub__ (self, other: Union[Variable, LinearCombination, Scalar]) -> LinearCombination:
        """Subtracts an element from the current element"""
        return self.__add__((-1.0) * other)

    # FIXME: This might introduce a bug
    def __rsub__ (self, other: Union[Variable, LinearCombination, Scalar]) -> LinearCombination:
        """Subtracts an element from the current element"""
        return self.__radd__((-1.0) * other)

    def __neg__ (self) -> LinearCombination:
        """Negates the variable, creating a linear combination."""
        return (-1.0) * self

    def __mul__ (self, other: Scalar) -> LinearCombination:
        """Multiples the variable by a scalar value, creating a linear combination."""
        if not isinstance(other, (int, float)): # NOTE: isinstance does not work with Union, so we cannot check the scalar type directly.
            raise TypeError(f"Variable {Variable.identifier} was multiplied by a non scalar value: {other}")
        
        return LinearCombination({self: other})

    def __rmul__ (self, other) -> LinearCombination:
        """Needed to enable multiplication."""
        return self.__mul__(other)

    def __hash__ (self):
        """Makes the variable hashable."""
        return hash(self.identifier)

    def __repr__ (self) -> str:
        return self.identifier

    def repr_with_constraints(self) -> str:
        """Returns a string representation of the variable and its constraints"""
        return (f"{self.lb} <= " if self.lb is not None else "") + self.identifier + (f" <= {self.ub}" if self.ub is not None else "") + " integral" if self.integral else ""
        


@dataclass
class LinearCombination:
    """Models a linear combination consisting of multiple variables with weights"""
    weights: Dict[Variable, Scalar] = field(default_factory=dict)

    def __add__ (self, other: Union[Variable, LinearCombination, Scalar]) -> LinearCombination:
        """Adds either a variable or a another linear combination to the linear combination"""
        if isinstance(other, (int, float)) and (other == 0.0 or other == 0): # NOTE: needed to enable the use of the sum function.
            return self

        if not isinstance(other, (Variable, LinearCombination)):
            raise TypeError(f"Linear Combination {self.__repr__()} was summed with something which was not neither a Variable or a another LinearCombination: {other}, which was of type {type(other)}")
            
        if isinstance(other, Variable):
            self.weights[other] = 1.0

        else:
            # NOTE: add the terms of the other linear combination to this one.
            for var in (self.weights.keys() | other.weights.keys()):
                self.weights[var] = self.weights.get(var, 0.0) + other.weights.get(var, 0.0)

        return self

    def __radd__ (self, other: Union[Variable, LinearCombination, Scalar]) -> LinearCombination:
        """Adds either a variable or a another linear combination to the linear combination"""
        return self.__add__(other)

    # FIXME: This might introduce a bug
    def __sub__ (self, other: Union[Variable, LinearCombination, Scalar]) -> LinearCombination:
        """Subtracts an element from the current element"""
        return self.__add__(-1.0 * other)

    # FIXME: This might introduce a bug
    def __rsub__ (self, other: Union[Variable, LinearCombination, Scalar]) -> LinearCombination:
        """Subtracts an element from the current element"""
        return self.__radd__(-1.0 * other)

    def __neg__ (self) -> LinearCombination:
        """Negates the variable, creating a linear combination."""
        return (-1.0) * self

    def __mul__ (self, other: Scalar) -> LinearCombination:
        """Multiples the variable by a scalar value, creating a linear combination."""
        if not isinstance(other, (int, float)): # NOTE: isinstance does not work with Union, so we cannot check the scalar type directly.
            raise TypeError(f"Linear Combination {self.__repr__()} was multiplied by a non scalar value: {other}")
        
        for (var, weight) in self.weights.items():
            self.weights[var] = other * weight
        
        return self

    def __rmul__ (self, other: Scalar) -> LinearCombination:
        """Multiples the variable by a scalar value, creating a linear combination."""
        return self.__mul__(other)
        

    def __repr__ (self) -> str:
        """Returns a string representation of the Linear Combination."""
        return "".join(f"{self.weights[var]}*{var.identifier} + " for var in sorted(self.weights.keys(), key=lambda var: var.identifier))[:-2]

@dataclass 
class Constraint:
    """Models a MILP constraint."""
    lc: LinearCombination
    rel: ComparisonOperator
    rhs: Scalar

    def __post_init__(self):
        """Convert any variable given as the linear combination to a linear combination"""
        if type(self.lc) == Variable:
            self.lc = LinearCombination({self.lc: 1.0})

    def __repr__ (self) -> str:
        """Returns a string representation of the constraint"""
        return self.lc.__repr__() + self.rel + " " + str(self.rhs)

@dataclass
class MILP:
    """Models a general MILP."""
    identifier: str
    vars: List[Variable]
    cons: List[Constraint]
    obj_fun: LinearCombination
    obj_sense: ObjectiveSense
    integral_mask: Vector = field(init=False)
    integral_indices: Vector = field(init=False)

    def __post_init__(self):
        """Initialize information about the indices of integral variables."""
        if type(self.obj_fun) == Variable:
            self.obj_fun = LinearCombination(weights={self.obj_fun: 1.0})
        
        self.integral_mask = np.array([var.integral for var in self.vars])
        self.integral_indices = np.array([i for i, var in enumerate(self.vars) if var.integral])

    def __repr__(self) -> str:
        """Returns a string representation of the MILP"""
        return f"Problem: {self.identifier}\n {self.obj_sense} {self.obj_fun} \n st: " + "\n     ".join(con.__repr__() for con in self.cons) + "\n     " + "\n     ".join(var.repr_with_constraints() for var in self.vars)

    def get_lower_bounds (self) -> Vector:
        """Gets the lower bounds of the variables""" 
        return np.array([var.lb for var in self.vars], dtype=np.double)

    def get_upper_bounds (self) -> Vector:
        """Gets the upper bounds of the variables""" 
        return np.array([var.ub for var in self.vars], dtype=np.double)

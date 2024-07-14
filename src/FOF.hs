-- | Contains code for storing first order logic formulas
module FOF where

type Id = String

data Term = Constant Id
          | Variable Id
          | Function Id [Term]
          deriving (Eq, Show)

data Quantifier = (:?:) | (:!:) deriving (Eq, Show)

data Operator = (:|:) | (:&:) | (:~&:) | (:~|:) | (:<~>:) deriving (Eq, Show)

data BinaryOperator = (:=>:) | (:<=>:) deriving (Eq, Show)

data FOF = Predicate Id [Term]
         | Negated FOF
         | Connected Operator [FOF]
         | Quantified Quantifier [Term] FOF
         | Binary BinaryOperator (FOF, FOF)
         | Equality (Term, Term)
         deriving (Eq, Show)

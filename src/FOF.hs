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
         | Quantified Quantifier [Id] FOF
         | Binary BinaryOperator (FOF, FOF)
         | Equality (Term, Term)
         deriving (Eq, Show)

-- | Remove syntactic sugar operator such as XOR, NAND, NOR from the formulas.
removeAuxiliaryOperators :: FOF -> FOF
removeAuxiliaryOperators fof = case fof of
  Connected (:<~>:) operands -> fof -- TODO:
  Connected (:~&:) operands -> Negated $ Connected (:|:) $ map removeAuxiliaryOperators operands
  Connected (:~|:) operands -> Negated $ Connected (:&:) $ map removeAuxiliaryOperators operands
  Connected operator operands -> Connected operator $ map removeAuxiliaryOperators operands
  Negated sub_fof -> Negated $ removeAuxiliaryOperators sub_fof
  Quantified quantifier vars sub_fof -> Quantified quantifier vars $ removeAuxiliaryOperators sub_fof
  Binary binary_operator (fst_fof, snd_fof) -> let
    fst_fof' = removeAuxiliaryOperators fst_fof
    snd_fof' = removeAuxiliaryOperators snd_fof
    in Binary binary_operator (fst_fof', snd_fof' )

renameVariableInTerm :: Id -> Id -> Term -> Term
renameVariableInTerm original_id new_id term = case term of
  (Variable id) -> if id == original_id then Variable new_id else Variable id
  (Function id' args) -> Function id' $ map (renameVariableInTerm original_id new_id) args
  _ -> term

renameVariable :: Id -> Id -> FOF -> FOF
renameVariable original_id new_id fof = case fof of
  (Connected op operands) -> Connected op $ map (renameVariable original_id new_id) operands
  (Binary bin_op (fst_fof, snd_fof)) -> Binary bin_op (renameVariable original_id new_id fst_fof, renameVariable original_id new_id snd_fof)
  (Negated sub_fof) -> renameVariable original_id new_id sub_fof
  (Equality (fst_term, snd_term)) -> Equality (renameVariableInTerm original_id new_id fst_term, renameVariableInTerm original_id new_id snd_term)
  (Predicate id args) -> Predicate id $ map (renameVariableInTerm original_id new_id) args

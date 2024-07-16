-- | Contains a set of functions for transforming first order formulas
module Prenex where

import Data.Tuple.Select

import FOF
-- REF: The functions contained within this file is based uppon the ideas found in https://www.csd.uwo.ca/~lkari/prenex.pdf
-- and http://www2.imm.dtu.dk/courses/02286/Slides/FirstOrderLogicPrenexFormsSkolemizationClausalFormTrans.pdf

-- | Removes every occurence of => and <=> an returns an equivalent FOF.
eliminateImplicationsAndEquivalences :: FOF -> FOF
eliminateImplicationsAndEquivalences fof = case fof of
  (Negated sub_fof) -> Negated $ eliminateImplicationsAndEquivalences sub_fof
  (Connected operator operands) -> Connected operator $ map eliminateImplicationsAndEquivalences operands
  (Quantified quantifier vars sub_fof) -> Quantified quantifier vars $ eliminateImplicationsAndEquivalences sub_fof
  (Binary (:=>:) (fst_fof, snd_fof)) -> let
    fst_fof' = eliminateImplicationsAndEquivalences fst_fof
    snd_fof' = eliminateImplicationsAndEquivalences snd_fof
    in Connected (:|:) [Negated fst_fof', snd_fof']
  (Binary (:<=>:) (fst_fof, snd_fof)) -> let
    fst_fof' = eliminateImplicationsAndEquivalences fst_fof
    snd_fof' = eliminateImplicationsAndEquivalences snd_fof
    in Connected (:&:) [Connected (:|:) [Negated fst_fof', snd_fof'], Connected (:|:) [fst_fof', Negated snd_fof']]
  formula -> formula

-- | Move negations inside all other logical connectives.
importNegations :: FOF -> FOF
importNegations fof = case fof of
  Negated (Negated sub_fof) -> importNegations sub_fof
  Negated (Quantified (:?:) vars sub_fof) -> Quantified (:!:) vars $ importNegations $ Negated sub_fof
  Negated (Quantified (:!:) vars sub_fof) -> Quantified (:?:) vars $ importNegations $ Negated sub_fof
  -- De Morgan's laws
  Negated (Connected (:&:) operands) -> Connected (:|:) $ map (importNegations . Negated) operands
  Negated (Connected (:|:) operands) -> Connected (:&:) $ map (importNegations . Negated) operands
  Connected operator operands -> Connected operator $ map importNegations operands
  formula -> formula -- stuff like predicates ect. NOTE: We have removed any implications and equivalences in the previous step.

-- | Standardize variables appart.
standardizeVariables :: FOF -> FOF -- FIXME: This function could be made much more efficent, using some elementary logic.
standardizeVariables formula = fst $ aux 0 formula where
  aux :: Integer -> FOF -> (FOF, Integer)
  aux counter fof = case fof of
    (Connected operation operands) -> (Connected operation $ fst pair, snd pair) where
      pair = foldl aux' ([], counter) operands where
        aux' (updated_operands, updated_counter) operand = (updated_operands ++ [fst pair'], updated_counter + snd pair') where
          pair' = aux updated_counter operand
    (Quantified quantifier vars sub_fof) -> let
      triple :: ([Id], FOF, Integer)
      triple = foldl aux' ([], sub_fof, counter) vars where
        aux' (updated_vars, updated_fof, updated_counter) original_id = (updated_vars ++ [show updated_counter], renameVariable original_id (show updated_counter) updated_fof, updated_counter + 1)
   
      pair :: (FOF, Integer)
      pair = aux (sel3 triple) (sel2 triple)
      in (Quantified quantifier (sel1 triple) (fst pair), snd pair)
    otherwise -> (fof, counter) -- NOTE: Basic stuff like predicates ect and negations which have been moved all the way inwards.

-- | Moves the quantifiers to the front of the formula.
pullQuantifiers :: FOF -> FOF
pullQuantifiers = id

-- | Converts a first order formula into prenex form
convertToPrenex :: FOF -> FOF
convertToPrenex = pullQuantifiers . standardizeVariables . importNegations . eliminateImplicationsAndEquivalences

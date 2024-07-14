-- | A Parser for the TPTP language (only for problems in CNF and FOF)

module TPTPParser where

import Data.Maybe
import FOF
import Data.Void
import Text.Megaparsec
import Text.Megaparsec.Char
import System.Exit

-- | Datatypes needed to store TPTP files
data TPTPFile = TPTPFile [TPTPStatement] TPTPStatus deriving (Eq, Show)

data TPTPStatus = Theorem | Unsaitisfiable | Satisfiable deriving (Eq, Show)

data TPTPStatement = TPTPStatement String TPTPType FOF deriving (Eq, Show)

data TPTPType = Axiom | Conjecture deriving (Eq, Show)

type Parser = Parsec Void String

-- | Parsing of terms
{-# INLINABLE parseAlphaNumWordStartingWith #-}
parseAlphaNumWordStartingWith :: Parser Char -> Parser String
parseAlphaNumWordStartingWith p = (:) <$> p <*> many (alphaNumChar <|> char '_')

parseTerm :: Parser Term
parseTerm = try parseVariable <|> try parseFunction <|> parseConstant

parseConstant :: Parser Term
parseConstant = Constant <$> parseAlphaNumWordStartingWith lowerChar

parseVariable :: Parser Term
parseVariable = Variable <$> parseAlphaNumWordStartingWith upperChar

parseFunction :: Parser Term
parseFunction = do
  name <- parseAlphaNumWordStartingWith lowerChar <* char '('
  arguments <- parseArguments <* char ')'
  return $ Function name arguments

{-# INLINABLE parseArguments #-}
parseArguments :: Parser [Term]
parseArguments = parseTerm `sepBy` char ','

-- | Parsing of FOF formulas
parsePredicate :: Parser FOF
parsePredicate = do
  name <- parseAlphaNumWordStartingWith lowerChar <* char '('
  arguments <- parseArguments <* char ')'
  return $ Predicate name arguments

-- | Parsing of quantified formulas
{-# INLINABLE parseVariableList #-}
parseVariableList :: Parser [Term]
parseVariableList = parseVariable `sepBy` char ','

parseQuantified :: Parser FOF
parseQuantified = try (aux "![" (:!:)) <|> aux "?[" (:?:) where
  aux :: String -> Quantifier -> Parser FOF
  aux prefix quantifier = do
    vars <- string prefix *> parseVariableList <* string "]:"
    Quantified quantifier vars <$> parseFormula

-- | Parsing of connected formulas
parseConnected :: Parser FOF
parseConnected = try (aux "&" (:&:))
                 <|> try (aux "|" (:|:))
                 <|> try (aux "~&" (:~&:))
                 <|> try (aux "~|" (:~|:))
                 <|> aux "<~>" (:<~>:) where
  aux seperator operator = do
    first_operand <- try parseFormula
    following_operands <- some (string seperator *> parseFormula)
    return $ Connected operator (first_operand:following_operands)

-- | Parse negated formulas
parseNegation :: Parser FOF
parseNegation = do
  f <- char '~' *> parseFormula
  return $ Negated f

-- | Parse Binary formulas
parseBinary :: Parser FOF
parseBinary = try (aux "=>" (:=>:)) <|> aux "<=>" (:<=>:) where
  aux :: String -> BinaryOperator -> Parser FOF
  aux seperator operator = do
    first_fof <- parseFormula <* string seperator
    second_fof <- parseFormula
    return $ Binary operator (first_fof, second_fof)

-- | Parse equality and not equals
parseEqualities :: Parser FOF
parseEqualities = try parseEquals <|> parseNotEquals where
  parseEquals = do
    first_term <- parseTerm <* char '='
    second_term <- parseTerm
    return $ Equality (first_term, second_term)
 
  parseNotEquals = do
    first_term <- parseTerm <* string "!="
    second_term <- parseTerm
    return $ (Negated . Equality) (first_term, second_term)

-- | Runs a parser between parenthesises.
parseParenthesised :: Parser a -> Parser a
parseParenthesised p = char '(' *> p <* char ')'

-- | Parses a first order formula.
parseFormula :: Parser FOF
parseFormula = try parseNegation
             <|> try parseEqualities
             <|> try parsePredicate
             <|> try parseQuantified
             <|> try (parseParenthesised parseEqualities)
             <|> try (parseParenthesised parseQuantified)
             <|> try (parseParenthesised parseConnected)
             <|> try (parseParenthesised parseBinary)
             <|> try (parseParenthesised parseNegation)
             <|> (parseParenthesised parsePredicate)

parseTPTPStatement :: Parser (Maybe TPTPStatement)
parseTPTPStatement = do
  name <- (string "fof(" <|> string "cnf(") *> parseAlphaNumWordStartingWith lowerChar <* char ','
  typ_string <- parseAlphaNumWordStartingWith lowerChar <* string ","
  formula <- parseFormula <* string ")."
  return $ case typ_string of
      typ | typ `elem` ["axiom", "hypothesis"] -> Just $ TPTPStatement name Axiom formula
      typ | typ `elem` ["conjecture"] -> Just $ TPTPStatement name Conjecture formula
      typ | typ `elem` ["negated_conjecture"] -> Just $ TPTPStatement name Conjecture $ Negated formula
      _ -> Nothing -- "Invalid TPTPObject type."

-- | Removes comments and white space.
preprocess :: String -> String
preprocess =  filter (`notElem` " \n\t") . removeComments where
  removeComments = aux False where
    aux True ('\n':ss) = '\n' : aux False ss
    aux True (_:ss) = aux True ss
    aux False ('%':ss) = aux True ss
    aux False (s:ss) = s : aux False ss
    aux _ [] = ""

-- | Parses all of the TPTP statements in the provied string.
parseTPTPStatements :: String -> String -> Either (ParseErrorBundle String Void) [Maybe TPTPStatement]
parseTPTPStatements source file_path = parse (many parseTPTPStatement <* eof) file_path $ preprocess source

-- | Parses a TPTP file, given its file path.
parseFile :: String -> IO () -- TPTPFile
parseFile file_path = do
  contents <- readFile file_path
  print $ "Preprocessed String: " ++ preprocess contents
  case parseTPTPStatements contents file_path of
      Left err -> die $ show err
      Right statements -> print $ TPTPFile (catMaybes statements) Theorem -- TODO: Read status from the file.

-- | Function for testing a parser
testParser :: (Show a) => Parser a -> String -> IO ()
testParser p source = print $ parse p "Test" $ preprocess source

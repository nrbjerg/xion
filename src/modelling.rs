use derivative::Derivative;
use std::collections::HashMap;

/// Models a MILP variable, which can either be integer valued or continuous
#[derive(Derivative)]
#[derivative(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Variable {
    pub id: String,
    #[derivative(PartialEq = "ignore", Hash = "ignore")]
    pub integer_valued: bool,
    #[derivative(PartialEq = "ignore", Hash = "ignore")]
    pub lb: Option<f64>,
    #[derivative(PartialEq = "ignore", Hash = "ignore")]
    pub ub: Option<f64>,
}

impl Variable {
    /// Creates a new continuous variable
    pub fn new_continous(id: &str, lb: Option<f64>, ub: Option<f64>) -> Self {
        Self {
            id: id.to_string(),
            integer_valued: false,
            lb: lb,
            ub: ub,
        }
    }

    /// Creates a new binary variable
    pub fn new_binary(id: &str) -> Self {
        Self {
            id: id.to_string(),
            integer_valued: true,
            lb: Some(0.0),
            ub: Some(1.0),
        }
    }

    /// Creates a new continuous variable
    pub fn new_integer(id: &str, lb: Option<f64>, ub: Option<f64>) -> Self {
        Self {
            id: id.to_string(),
            integer_valued: true,
            lb: lb,
            ub: ub,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LinearCombination {
    pub terms: HashMap<Variable, f64>,
}

impl LinearCombination {
    /// Adds a term to the linear combination, i.e. the term coef * var
    pub fn add_term(&mut self, coef: f64, var: &Variable) {
        self.terms.insert(var.clone(), coef);
    }

    pub fn negate_terms(&mut self) {
        for coef in self.terms.values_mut() {
            *coef *= -1.0;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OptimizationSense {
    /// Specifies wether we have a maximization or minimization problem.
    MIN,
    MAX,
}

#[derive(Debug, Clone)]
pub struct Objective {
    pub sense: OptimizationSense,
    pub linear_combination: LinearCombination,
}

#[derive(Debug, Clone, Copy)]
pub enum Comparison {
    /// The sense of an constraint i.e. <= = or >=.
    LEQ, // <=
    GEQ, // >=
    EQ,  // =
}

#[derive(Debug, Clone)]
pub struct Constraint {
    /// Models a linear constraint
    pub id: String,
    pub sense: Comparison,
    pub linear_combination: LinearCombination,
    pub rhs: f64, // Right hand side of the constraint
}

pub struct Model {
    /// Models a MILP problem, with a set of variables, constraints and a singular objective
    pub id: String,
    pub objective: Objective,
    pub variables: Vec<Variable>,
    pub constraints: Vec<Constraint>,
}

impl Model {
    /// Creates an empty model, which can later be defined in details.
    pub fn new(
        id: &str,
        objective: Objective,
        variables: Vec<Variable>,
        constraints: Vec<Constraint>,
    ) -> Self {
        Self {
            id: id.to_string(),
            variables: variables,
            constraints: constraints,
            objective: objective,
        }
    }

    /// Adds a variable to the model
    pub fn add_variable(&mut self, var: Variable) {
        self.variables.push(var);
    }

    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }
}

/// Setup a simple knapsack problem for demonstration purposes.
pub fn setup_knapsack_problem() -> Model {
    let xs: Vec<Variable> = (0..8)
        .map(|i| Variable::new_binary(&format!("x{:?}", i)))
        .collect();
    let ws: Vec<f64> = vec![0.15, 0.21, 0.24, 0.39, 0.78, 0.12, 0.08, 0.54];
    let vs: Vec<f64> = vec![0.21, 0.32, 0.16, 0.35, 0.91, 0.21, 0.04, 0.43];
    let w: f64 = 1.0;

    let objective = Objective {
        linear_combination: LinearCombination {
            terms: xs.iter().cloned().zip(vs.into_iter()).collect(),
        },
        sense: OptimizationSense::MAX,
    };

    let maximal_weight_constraint = Constraint {
        id: "Maximal Weight Constraint".to_string(),
        linear_combination: LinearCombination {
            terms: xs.iter().cloned().zip(ws.into_iter()).collect(),
        },
        rhs: w,
        sense: Comparison::LEQ,
    };

    Model::new("Knapsack", objective, xs, vec![maximal_weight_constraint])
}

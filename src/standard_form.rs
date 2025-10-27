use crate::modelling::{self, Comparison, Constraint, LinearCombination, Variable};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use std::iter::zip;

/// Models a MILP in standard form that is min c^Tx subject to Ax = b, b >= 0 and x >= 0.
#[allow(non_snake_case)]
#[derive(Debug, Clone)]
pub struct StandardForm {
    cm: DMatrix<f64>,  // Constraint Matrix (A)
    rhs: DVector<f64>, // Right hand side of equations (b)
    c: DVector<f64>,   // coefficients of objective function
}

impl StandardForm {
    /// Converts the supplied model to a standard form, which can be used within simplex
    pub fn generate(mut model: modelling::Model) -> Self {
        // 1. Construct coefficient matrix cm (A) and rhs (b)
        // NOTE: this may add variables, hence it needs to be done first
        let mut constraints = model.constraints.clone();
        let mut variables = model.variables.clone();

        // 1.1 Add the extra constraints from the bounds on the variables
        // FIXME: Check that all of the appropriate cases are handled.
        for var in variables.iter() {
            match var.lb {
                Some(0.0) => {} // NOTE: Special case where we do not need to introduce an extra constraint & slack variable.
                Some(var_min) => {
                    // In this case we need to add the constraint to the model
                    model.add_constraint(Constraint {
                        id: format!("LB for {}", var.id),
                        linear_combination: LinearCombination {
                            terms: HashMap::from([(var.clone(), 1.0)]),
                        },
                        sense: Comparison::GEQ,
                        rhs: var_min,
                    });
                }
                None => {} // No lower bound => no constraint.
            }

            if let Some(var_max) = var.ub {
                model.add_constraint(Constraint {
                    id: format!("LB for {}", var.id),
                    linear_combination: LinearCombination {
                        terms: HashMap::from([(var.clone(), 1.0)]),
                    },
                    sense: Comparison::LEQ,
                    rhs: var_max,
                });
            }
        }

        for (i, constraint) in constraints.iter().enumerate() {
            // 1.1 Add slack variables if necessary
            match constraint.sense {
                Comparison::GEQ => {
                    let slack_variable =
                        Variable::new_continous(&format!("@s{:?}", i), Some(0.0), None);
                    constraint
                        .linear_combination
                        .add_term(-1.0, &slack_variable);
                    variables.push(slack_variable);
                }
                Comparison::LEQ => {
                    let slack_variable =
                        Variable::new_continous(&format!("@s{:?}", i), Some(0.0), None);
                    constraint.linear_combination.add_term(1.0, &slack_variable);
                    variables.push(slack_variable);
                }
                Comparison::EQ => {}
            };
            // 1.2 Ensure that rhs (b) is non-negative
            if constraint.rhs < 0.0 {
                constraint.linear_combination.negate_terms();
            }
            constraints[i] = *constraint;
        }

        // TODO: 1.X convert preprocessed constraints to coefficient matrix cm (A) and rhs (b)
        let mut cm = DMatrix::zeros(constraints.len(), variables.len());
        for (i, constraint) in constraints.iter().enumerate() {
            todo!()
        }

        // 2. Convert objective to standard form i.e. min c^T x
        let mut coefficient_hashmap = HashMap::new();
        for var in model.objective.linear_combination.terms.keys() {
            let coef = *model.objective.linear_combination.terms.get(var).unwrap();
            coefficient_hashmap.insert(
                var,
                if model.objective.sense == modelling::OptimizationSense::MAX {
                    -coef
                } else {
                    coef
                },
            );
        }
        let c = DVector::from_iterator(
            model.variables.len(),
            model
                .variables
                .iter()
                .map(|var| coefficient_hashmap.get(var).unwrap_or(&0.0))
                .collect(),
        );

        Self { cm: cm, b: b, c: c }
    }
}

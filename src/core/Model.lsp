(in-package :model)

(defstruct milp-variable
    id
    lower-bound
    upper-bound
    integer?
    value)

(defstruct milp-constraint
    id
    terms ; a list of (coefficeint . variable)
    sense ; :<= := or :>=
    right-hand-side)

(defstruct milp-model
    id 
    objective-terms ; a list of (coefficient . variable)
    objective-sense ; :min or :max
    variables
    constraints
    best-solution
    best-value)

;; ---------------------------------------- Setup of models -------------------------------------
(defun create-milp-model (&key id)
    "Initializes an empty MILP model, with a given objective sense."
    (make-milp-model :id id
                     :variables '()
                     :objective-terms '()
                     :constraints '()
                     :objective-sense nil
                     :best-solution nil
                     :best-value nil))

(defun add-milp-variable (model &key id (lower-bound 0.0) 
                                (upper-bound most-positive-double-float)
                                (integer? nil))
    "Adds a variable to the model, and performs some rudamentary checks on it."
    (when (find id (milp-model-variables model) ; Check that the variable is not already defiend
        :key #'milp-variable-id :test #'eq)
        (error "Variable ~A already defined in model." id))
    (when (> lower-bound upper-bound) ; Check that the bounds are consistent
        (error "Lower bound ~A cannot be greater than upper bonund ~A for variable ~A" lower-bound upper-bound id))
    (let ((new-variable (make-milp-variable 
        :id id
        :lower-bound lower-bound
        :upper-bound upper-bound
        :integer? integer?
        :value nil)))
        (push new-variable (milp-model-variables model))
        new-variable))

(defun add-milp-constraint (model &key id terms sense right-hand-side) 
    "Adds a constraint to the model"
    (let ((new-constraint (make-milp-constraint :id id
                                                :terms terms
                                                :sense sense
                                                :right-hand-side right-hand-side)))
    (push new-constraint (milp-model-constraints model))
    new-constraint))

(defun add-milp-objective (model &key objective-sense terms)
    "Defines the objective for a model"
    (progn
        (setf (milp-model-objective-sense model) objective-sense)
        (setf (milp-model-best-value model) (if (eq objective-sense :min)
                                                most-positive-double-float
                                                most-negative-double-float))
        (setf (milp-model-objective-terms model) terms))
    model)
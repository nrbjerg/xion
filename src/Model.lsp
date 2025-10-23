(in-package :xion)

(defstruct $variable
    id
    lower-bound
    upper-bound
    coefficient-in-objective
    integer?
    value)

(defstruct $constraint
    id
    terms ; a list of (variable . coefficient)
    sense ; :<= := or :>=
    right-hand-side)

(defstruct $model
    objective-terms ; a list of (variable . coefficient)
    objective-sense ; :min or :max
    variables
    constraints
    best-solution
    best-value)

;; ---------------------------------------- Setup of models -------------------------------------
(defun create-model (&key (objective-sense :max))
    "Initializes an empty MILP model, with a given objective sense."
    (make-$model :variables '()
                 :objective-terms '()
                 :constraints '()
                 :objective-sense objective-sense
                 :best-solution nil
                 :best-value (if (eq objective-sense :max)
                                most-negative-double-float
                                most-positive-double-float)))

(defun add-variable (model &key id (lower-bound 0.0) (upper-bound most-positive-double-float)
                                (coefficient-in-objective 0.0) (integer? nil))
    "Adds a variable to the model, "
    (when (find id ($model-variables model) ; Check that the variable is not already defiend
        :key #'$variable-id :test #'eq)
        (error "Variable ~A already defined in model." id))
    (when (> lower-bound upper-bound) ; Check that the bounds are consistent
        (error "Lower bound ~A cannot be greater than upper bonund ~A for variable ~A" lower-bound upper-bound id))
    (let ((new-variable (make-$variable 
        :id id
        :lower-bound lower-bound
        :upper-bound upper-bound
        :coefficient-in-objective coefficient-in-objective
        :integer? integer?
        :value nil)))
        (push new-variable ($model-variables model))
        model))
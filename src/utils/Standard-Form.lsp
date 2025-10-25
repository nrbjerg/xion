(in-package :standard-form)

(defun negate-linear-combination (terms)
    "Negates in the terms in a linear combination (i.e. a list of cons cells of the form (coef . var))"
    (loop :for (coef . var) :in terms :collect (cons (- coef) var)))

; NOTE: We do not keep track of this, so the objective values may be weird, however I digress.
(defun convert-objective-to-minimization (model) 
    "converts the models objective to a minimization problem."
    (when (eq (model:milp-model-objective-sense model) :max)
        (progn 
            (setf (model:milp-model-objective-terms model) (negate-linear-combination (model:milp-model-objective-terms model))) 
            (setf (model:milp-model-objective-sense model) :min)))
    model)

(defun convert-inequality-constraints-to-equality-constraints (model)
    "Update the constraints to be in standard form."
    (loop :for const :in (model:milp-model-constraints model)
          :for i :from 0
          :do (progn
            ; Convert all inequalities of the form a^Tx >= b to -a^Tx <= -b
            (when (eq (model:milp-constraint-sense const) :>=)
                (progn (setf (model:milp-constraint-right-hand-side const) (- (model:milp-constraint-right-hand-side const)))
                       (setf (model:milp-constraint-terms const) (negate-linear-combination (model:milp-constraint-terms const)))
                       (setf (model:milp-constraint-sense const) :<=)))
            ; Introduce slact variables and 
            (when (eq (model:milp-constraint-sense const) :<=)
                (let* ((surplus-variable (model:add-milp-variable model :id (format t "$s~d" i)))
                       (term (if (< (model:milp-constraint-right-hand-side const) 0) 
                            (cons -1.0 surplus-variable)
                            (cons 1.0 surplus-variable))))
                       (push term (model:milp-constraint-terms const))))))
    model)

(defun convert-model-to-standard-form (model)
    "covnerts the model to standard form"
    (convert-inequality-constraints-to-equality-constraints (convert-objective-to-minimization model)))


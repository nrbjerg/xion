(in-package :simplex)

(defun solve-lp-relaxation (model)
    "Solve the LP relaxation of MODEL using simplex and return (variable-values-in-solution objective-value status)"
    (values (mapcar (lambda (var) (cons (model:$variable-id var) (model:$variable-lower-bound var)))
            (model:$model-variables model))
    0.0
    :optimal))


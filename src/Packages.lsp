(in-package :cl)

(defpackage :model
    (:use :cl)
    (:export :create-model :add-variable :add-constraint :finalize-model
     :variable-id :variable-lower-bound :variable-upper-bound 
     :variable-coefficient-in-objective :variable-integer? :variable-value 
     :constraint-id :constraint-terms
     :constraint-sense :constraint-right-hand-side
     :model-objective-terms :model-objective-sense :model-variables 
     :model-constraints :model-best-solution :model-best-value))

(defpackage :standard-form
    (:use :cl :model)
    (:export :convert-model-to-standard-form))

(defpackage :simplex
    (:use :cl :model)
    (:export :solve-LP-relaxation))

(defpackage :xion
    (:use :cl)
    (:export :create-model :add-variable :add-constraint :finalize-model :solve-milp))

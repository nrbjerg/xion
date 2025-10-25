(defpackage :knapsack
    (:use :cl :xion))

(in-package :knapsack)

; NOTE: define parameters such as weights etc.
(defparameter *ws* (list 3.2 1.5 2.4 6.0))
(defparameter *W* 20.0)
(defparameter *vs* (list 2.4 0.9 1.3 4.8))

; NOTE: setup the model
(defparameter *model* (model:create-milp-model :id "Constrained Knapsack Problem"))
(defparameter *xs* (loop :for i :below 4
                         :for upper-bound :in (list 2 3 4 1)
                         :collect (add-milp-variable *model* :id (format nil "x~d" i)
                                                             :upper-bound upper-bound
                                                             :integer? t)))

(model:add-milp-constraint *model* :id "weight constraint" 
                                   :terms (mapcar (lambda (w x) (cons w x)) *ws* *xs*)
                                   :sense :<=
                                   :right-hand-side *W*)

(model:add-milp-objective *model* :objective-sense :max 
                                  :terms (mapcar (lambda (v x) (cons v x)) *vs* *xs*))
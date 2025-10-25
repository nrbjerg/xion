(defsystem "xion"
  :description "A MILP solver, created for educational purposes."
  :author "Martin Sig Nørbjerg <martin1509@outlook.dk>"
  :license "MIT"
  :version "0.0.1"
  :depends-on (#:lla)
  :serial t
  :components ((:module "src"
                :components
                ((:file "Packages")
                 (:file "Model")
                 (:file "utils/Standard-Form")
                 (:file "Simplex")
                 (:file "Main")))))
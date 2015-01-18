Neural net library I made while working on https://karpathy.github.io/neuralnets/

Make networks.
```lisp
(defparameter *net-1*
  (make-product-gate
   (make-product-gate (make-variable-gate -2)
		              (make-variable-gate 3))
   (make-product-gate (make-variable-gate 2)
		              (make-variable-gate 1))))
```

Use Them.
```lisp
; Get the value.
(forward *net-1*) ; => -12
; Backpropogate
(backward *net-1* 1 0.0001) ; => nil
(forward *net-1*) ; => ~-12
```

Variable gates change when back propgated, constant gates do not. Constant gates should be used for input values.
```lisp
(make-variable-gate -2) ; Back propogate me and I'll change
(make-constant-gate -2) ; Back propogate me and I won't do anything
```

To change an input value, just change the value in a constant gate manually.
```lisp
(defvar *input-gate-1* (make-constant-gate 5))
(setf (gate-value *input-gate-1*) 7)
```

Still working on a good way to build trees and retain references to constant and variable gates. This is what I've come up with so far.
```lisp
;; Returns three values; The entire neural net, a list of constant gates
;; and a list of variable gates.
(net-with-holes
  (make-sum-gate*
  (make-product-gate (make-variable-gate 1) (make-constant-gate nil))
  (make-product-gate (make-variable-gate -2) (make-constant-gate nil))
  (make-variable-gate -1)))
```
I don't like it that much.



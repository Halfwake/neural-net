;;;; neural-net.lisp

(in-package #:neural-net)

;;; "neural-net" goes here. Hacks and glory await!

(defclass di-gate ()
  ((input-a :initarg :a
	    :reader input-a)
   (input-b :initarg :b
	    :reader input-b))
  (:documentation "Represents a neural net gate with two inputs."))

(defgeneric backward (gate gradient rate)
  (:documentation "Back propogate gradients."))

(defgeneric forward (gate)
  (:documentation "Return the output of a gate."))


(defclass value-gate ()
  ((variable-value :initarg :value
		   :accessor variable-value))
  (:documentation "A gate that uses a value for input."))

(defmethod forward ((gate value-gate))
  (variable-value gate))

(defclass variable-gate (value-gate) ()
  (:documentation "A variable value input gate."))

(defun make-variable-gate (value)
  (make-instance 'variable-gate :value value))

(defmethod backward ((gate variable-gate) gradient rate)
  (incf (variable-value gate) (* gradient rate))
  t)


(defclass constant-gate (value-gate) ()
  (:documentation "A constant value input gate."))

(defun make-constant-gate (value)
  (make-instance 'constant-gate :value value))

(defmethod backward ((gate constant-gate) gradient rate)
  (declare (ignore gate gradient rate))
  t)

(defclass product-gate (di-gate) ()
  (:documentation "A di-gate that outputs the product of its two inputs."))

(defun make-product-gate (a b)
  (make-instance 'product-gate :a a :b b))

(defmethod backward ((gate product-gate) gradient rate)
  (let ((value-a (forward (input-a gate)))
	(value-b (forward (input-b gate))))
    (backward (input-a gate) (* value-b gradient) rate)
    (backward (input-b gate) (* value-a gradient) rate)))

(defmethod forward ((gate product-gate))
  (* (forward (input-a gate))
     (forward (input-b gate))))

(defun make-product-gate* (&rest input-gates)
  (reduce #'make-product-gate input-gates))


(defclass sum-gate (di-gate) ()
  (:documentation "A di-gate that outputs the sum of its two inputs."))

(defun make-sum-gate (a b)
  (make-instance 'sum-gate :a a :b b))

(defmethod backward ((gate sum-gate) gradient rate)
  (backward (input-a gate) gradient rate)
  (backward (input-b gate) gradient rate))

(defmethod forward ((gate sum-gate))
  (+ (forward (input-a gate))
     (forward (input-b gate))))

(defun make-sum-gate* (&rest input-gates)
  (reduce #'make-sum-gate input-gates))


(defclass max-gate (di-gate) ()
  (:documentation "A di-gate that outputs the max of its two inputs."))

(defun make-max-gate (a b)
  (make-instance 'max-gate :a a :b b))

(defmethod backward ((gate max-gate) gradient rate)
  (let* ((gate-a (input-a gate))
	 (gate-b (input-b gate))
	 (value-a (forward gate-b))
	 (value-b (forward gate-a)))
    ;; TODO What if they're equal? Randomly switch between them?
    (cond ((> value-a value-b)
	   (backward gate-a gradient rate)
	   (backward gate-b 0 rate))
	  ((<= value-a value-b)
	   (backward gate-a 0 rate)
	   (backward gate-b gradient rate)))))

(defmethod forward ((gate max-gate))
  (let ((value-a (forward (input-a gate)))
	(value-b (forward (input-b gate))))
    (if (> value-a value-b)
	value-a
	value-b)))

(defun make-max-gate* (&rest input-gates)
  (reduce #'make-max-gate input-gates))


(defclass uni-gate ()
  ((input-v :initarg :input-v
	    :reader input-v))
  (:documentation "Represents a neural net with one input."))


(defclass sigmoid-gate (uni-gate) ()
  (:documentation "A uni-gate that outputs the sigmoid of its one input."))

(defun make-sigmoid-gate (input-v)
  (make-instance 'sigmoid-gate :input-v input-v))

(defun sigmoid (x)
  (/ 1
     (1+ (expt (exp 1)
	       (- x)))))

(defmethod forward ((gate sigmoid-gate))
  (sigmoid (forward (input-v gate))))

(defmethod backward ((gate sigmoid-gate) gradient rate)
  (let* ((value-v (sigmoid (forward (input-v gate))))
	 (local-gradient (* value-v (- 1 value-v))))
    (backward (input-v gate) (* local-gradient gradient) rate)))


(defparameter *net-1*
  (make-product-gate
   (make-product-gate (make-variable-gate -2)
		      (make-variable-gate 3))
   (make-product-gate (make-variable-gate 2)
		      (make-variable-gate 1))))

(defparameter *net-2*
  (make-product-gate
   (make-sum-gate (make-variable-gate -2)
		  (make-variable-gate 5))
   (make-variable-gate -4)))

(defparameter *net-3*
  (make-sigmoid-gate
   (make-sum-gate*
    (make-product-gate
     (make-variable-gate 1)
     (make-variable-gate -1))
    (make-product-gate
     (make-variable-gate 2)
     (make-variable-gate 3))
    (make-variable-gate -3))))

(defparameter *net-4*
  (make-product-gate*
   (make-variable-gate 2)
   (make-variable-gate 3)
   (make-variable-gate 3)))

(defparameter *net-5*
  (make-sum-gate*
   (make-variable-gate 1)
   (make-variable-gate 3)
   (make-variable-gate 5)))

(defparameter *net-6*
  (make-max-gate
   (make-variable-gate 1)
   (make-variable-gate 2)))

;(defun training-protocol (x y)
 ; (make-sum-gate*


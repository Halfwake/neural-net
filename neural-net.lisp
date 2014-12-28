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


(defclass constant-gate ()
  ((constant-value :initarg :value
		   :accessor constant-value))
  (:documentation "A constant input gate."))

(defun make-constant-gate (value)
  (make-instance 'constant-gate :value value))

(defmethod backward ((gate constant-gate) gradient rate)
  (incf (constant-value gate) (* gradient rate))
  t)

(defmethod forward ((gate constant-gate))
  (constant-value gate))


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


(defparameter *net-1*
  (make-product-gate
   (make-product-gate (make-constant-gate -2)
		      (make-constant-gate 3))
   (make-product-gate (make-constant-gate 2)
		      (make-constant-gate 1))))

(defparameter *net-2*
  (make-product-gate
   (make-sum-gate (make-constant-gate -2)
		  (make-constant-gate 5))
   (make-constant-gate -4)))

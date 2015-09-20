(ns clj-machine-learning.matrices
  (:require [clojure.core.matrix :as m]
            [clatrix.core :as cl]
            [incanter.charts :refer [xy-plot add-points]]
            [incanter.core :refer [view]]))

;; (m/identity-matrix n) - generate identity matrix
;; (cl/rnorm) - gererate matrix of random elements

(defn create []
  (cl/matrix [[1 2 3] [4 5 6]]))

(defn lmatrix [n]
  (m/compute-matrix :clatrix [n (+ n 2)]
                    (fn [i j] ({0 -1 1 2 2 -1} (- j i) 0))))

(defn problem
  "Return a map of the problem setup for a given matrix size,
  number of observed values and regularization parameter"
  [n n-observed lambda]
  (let [i (shuffle (range n))]
    {:L (m/mmul (lmatrix n) lambda)
     :observed (take n-observed i)
     :hidden (drop n-observed i)
     :observed-values (m/matrix :clatrix (repeatedly n-observed rand))}))

(defn solve
  "Return a map containing the approximated value y of each hidden point x"
  [{:keys [L observed hidden observed-values] :as problem}]
  (let [nc (m/column-count L)
        nr (m/row-count L)
        L1 (cl/get L (range nr) hidden)
        L2 (cl/get L (range nr) observed)
        l11 (m/mmul (m/transpose L1) L1)
        l12 (m/mmul (m/transpose L1) L2)]
    (assoc problem :hidden-values 
           (m/mmul -1 (m/inverse l11) l12 observed-values))))

(defn plot-points
  "Plots sample points of a solution s"
  [s]
  (let [X (concat (:hidden s) (:observed s))
        Y (concat (:hidden-values s) (:observed-values s))]
    (view
      (add-points
        (xy-plot X Y) (:observed s) (:observed-values s)))))

(defn plot-rand-sample []
  (plot-points (solve (problem 150 10 30))))

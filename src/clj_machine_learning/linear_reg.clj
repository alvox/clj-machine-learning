(ns clj-machine-learning.linear-reg
  (:require [clatrix.core :as cl]
            [incanter.charts :refer [scatter-plot add-lines]]
            [incanter.core :refer [view]]
            [incanter.stats :refer [linear-model]]))

(def x (cl/matrix [8.401 14.475 12.396 13.127 5.004 8.339 15.692 17.108 9.235 12.029]))
(def y [-1.57 2.32 0.424 0.814 -2.3 0.01 1.954 2.269 -0.635 0.328])

(defn show-plot []
  (view (scatter-plot x y)))

(defn show-linear-model []
  (view (add-lines (scatter-plot x y) x (:fitted (linear-model y x)))))

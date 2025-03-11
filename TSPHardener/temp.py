import core.generate_tsp as gt
import core.mutate_tsp as mt
from icecream import ic

matrix = gt.generate_tsp(4, "euclidean", "uniform", 1.2).round(2)
ic(matrix)

ic(mt.swap("euclidean", matrix))

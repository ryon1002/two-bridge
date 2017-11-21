import math
import itertools
import numpy as np
from scipy.spatial.distance import squareform, pdist


class CalcPredictabilityPath(object):
    def calc_optimal_path(self, start, points):
        points = np.array(points)
        dists = squareform(pdist(points, metric="cityblock"))
        start_dists = np.sum(np.abs(points - start), axis=1)

        min_path = self._calc_optimal_path(start_dists, dists, range(len(points)))
        print points[list(min_path)]

    def _calc_optimal_path(self, start_dists, dists, valid_points):
        min_path = None
        min_len = 10000
        lens = np.empty(math.factorial(len(valid_points)))
        for n, path in enumerate(itertools.permutations(valid_points)):
            path_len = start_dists[path[0]] + sum([dists[path[i], path[i + 1]] for i in range(len(path) - 1)])
            lens[n] = -path_len
            if path_len < min_len:
                min_len = path_len
                min_path = path
        lens += min_len
        return min_path, min_len, 1 / np.sum(np.exp(lens))

    def calc_1_predictive_path(self, start, points):
        points = np.array(points)
        dists = squareform(pdist(points, metric="cityblock"))
        start_dists = np.sum(np.abs(points - start), axis=1)
        for i in itertools.permutations(range(len(points)), 1):
            valid_points = [p for p in range(len(points)) if p not in set(i)]
            path, plan, prob = self._calc_optimal_path(dists[i[-1]], dists, valid_points)
            print points[list(path)], plan, prob

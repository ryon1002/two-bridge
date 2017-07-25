import numpy as np
from core import MDP


class GridWorldMDPBase(MDP.MDP):
    dirActionIdx = {"^": np.array([1, 0]), "<": np.array([0, -1]), ">": np.array([0, 1]), "v": np.array([-1, 0]),
                    "o": np.array([0, 0])}

    def __init__(self, x_num, y_num, prob=1.0, err_prob=0.0, **kwargs):
        self.aMap = {0: "^", 1: "<", 2: ">", 3: "v", 4: "o"}
        self.shape = (y_num, x_num)
        MDP.MDP.__init__(self, y_num * x_num, len(self.aMap), **kwargs)
        a_t = np.ones((self.a, self.a)) * err_prob
        np.fill_diagonal(a_t, prob)
        for s in range(self.s):
            nexts = [self.get_next_state_for_dir_action(s, self.aMap[a]) for a in range(self.a)]
            for i, ns in enumerate(nexts):
                self.t[:, s, ns] += a_t[:, i]

    def get_next_state_for_dir_action(self, s, dir_action):
        next_pos = np.unravel_index(s, self.shape) + GridWorldMDPBase.dirActionIdx[dir_action]
        return np.ravel_multi_index(next_pos, self.shape, mode="clip")

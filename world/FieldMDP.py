import numpy as np
import itertools
import GridWorldMDPBase


class FieldMDP(GridWorldMDPBase.GridWorldMDPBase):
    def __init__(self, world_map, **kwargs):
        self.key_a_map = {"up": 0, "down": 3, "right": 2, "left": 1}
        self.map = world_map
        GridWorldMDPBase.GridWorldMDPBase.__init__(self, self.map.shape[1], self.map.shape[0], 1, 0, **kwargs)
        self.f_map = self.map.flatten()[::-1]

        for a, s in itertools.product(range(4), range(self.s)):
            if self.f_map[np.argmax(self.t[a, s, :])] == 1:
                self.t[a, s, :] = 0
                self.t[a, s, s] = 1
        self.r[:4, :, :] = -0.01

    def set_goal(self, g):
        self.set_terminate_states(g)
        self.r[:, :, g] = 1
        self.r[:, g, g] = 0

    def make_road(self, s, reward=-0.01):
        pos = np.array(np.unravel_index(s, self.shape))
        for offset in np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]):
            s2 = np.ravel_multi_index(pos + offset, self.shape, mode="clip")
            if self.f_map[s2] == 0:
                self.connect(s, s2, reward)

    def connect(self, s1, s2, reward):
        s1y, s1x = np.unravel_index(s1, self.shape)
        s2y, s2x = np.unravel_index(s2, self.shape)
        if abs(s1y - s2y) == 1 and s1x == s2x:
            a12, a21 = (3, 0) if s1y - s2y == 1 else (0, 3)
        elif abs(s1x - s2x) == 1 and s1y == s2y:
            a12, a21 = (1, 2) if s1y - s2y == 1 else (2, 1)
        self.t[a12, s1, s1] = 0
        self.t[a12, s1, s2] = 1
        self.r[a12, s1, s2] = reward
        self.t[a21, s2, s2] = 0
        self.t[a21, s2, s1] = 1
        self.r[a21, s2, s1] = reward

    def shrink_state(self, start_state):
        reachable_state = self.calc_reachable_state(start_state)
        self.shrinkMap = {s: n for n, s in enumerate(reachable_state)}
        self.i_shrinkMap = {n: s for n, s in enumerate(reachable_state)}
        self.s = len(reachable_state)
        self.t = np.array([[self.t[a, s, reachable_state] for s in reachable_state] for a in range(self.t.shape[0])])
        self.r = np.array([[self.r[a, s, reachable_state] for s in reachable_state] for a in range(self.r.shape[0])])

    def calc_reachable_state(self, start_state):
        open_states = set([start_state])
        result = set()
        while len(open_states) > 0:
            s = open_states.pop()
            result.add(s)
            next_states = set(np.where(np.sum(self.t[:, s, :], axis=0))[0])
            next_states.difference_update(open_states)
            next_states.difference_update(result)
            open_states.update(next_states)
        return np.array(sorted(result))

    def print_map(self):
        s_map = np.zeros_like(self.map, dtype=object)
        for s in range(self.s):
            y, x = np.unravel_index(s, self.shape)
            s_map[self.shape[0] - y - 1, x] = (s, int(self.map[self.shape[0] - y - 1, x]))
        print s_map

import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from world import GridWorldMDPBase
from world.core import MDP
from solver import ValueIteration


class HierarchicalItemPickWorld(MDP.MDP):
    def __init__(self):
        self.map = np.zeros((10, 10))
        # self.items = {(2, 0): 1, (4, 2): 2, (3, 4): 2}
        self.items = {0: (2, 0), 1: (4, 2), 2: (3, 4), 3: (2, 5)}
        base_mdp = GridWorldMDPBase.GridWorldMDPBase(self.map.shape[1], self.map.shape[0], 1, 0, d=1)
        self.base_s = base_mdp.s
        self.mdp_num = 2 ** len(self.items) - 1
        self.item_index = np.array([2 ** i for i in range(len(self.items))])
        self.solver = ValueIteration.ValueIteration()

        # super(MultipleItemPickWorld, self).__init__(base_mdp.s * self.mdp_num + 1, base_mdp.a)
        # for flags in itertools.product([True, False], repeat=len(self.items)):
        #     flags = np.array(flags)
        #     if not np.sum(flags):
        #         continue
        #     mdp_idx = self.calc_mdp_index(flags)
        #     self.t[:, base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1),
        #     base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1)] = base_mdp.t
        #     for i in range(len(flags)):
        #         if flags[i]:
        #             target = np.ravel_multi_index(self.items[i], self.map.shape)
        #             to_flags = flags.copy()
        #             to_flags[i] = False
        #             to_mdp_idx = self.calc_mdp_index(to_flags)
        #             if not np.sum(to_flags):
        #                 self.t[:, base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1), -1] = \
        #                     self.t[:, base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1), base_mdp.s * mdp_idx + target]
        #             else:
        #                 self.t[:, base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1), base_mdp.s * to_mdp_idx + target] = \
        #                     self.t[:, base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1), base_mdp.s * mdp_idx + target]
        #             self.t[:, base_mdp.s * mdp_idx:base_mdp.s * (mdp_idx + 1), base_mdp.s * mdp_idx + target] = 0
        # self.set_terminate_states(self.s - 1)

        self.r[:, :, :] = -0.01
        self.r[:, :, -1] = 1

    def calc_mdp_index(self, items):
        return self.mdp_num - np.dot(items, self.item_index)

    def calc_policy(self):
        return self.solver.get_greedy_policy(self)

    def set_reward(self, reward):
        # r_map = {0: -2, 1: -0.01, 2: -0.1}
        # r_map = {0: -0.5, 1: -0.2, 2: -0.01}
        for s in range(self.s):
            self.r[:, :, s] = reward[self.map[np.unravel_index(s, self.map.shape)]]

        r_states = np.ravel_multi_index(np.where(self.item == 1), self.shape)
        for r_state in r_states:
            self.set_goal(r_state)

    def set_goal(self, g):
        self.set_terminate_states(g)
        self.r[:, :, g] = 1
        self.r[:, g, g] = 0

    def show_world(self, path=()):
        for y, x in itertools.product(*[range(i) for i in self.map.shape]):
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))
            # plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color_map[self.map[y, x]], edgecolor="k"))
            # if self.item[y, x] == 1:
            #     plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor="gold", edgecolor="k"))

        color_map = {1: "b", 2: "g"}
        for (y, x) in self.items.viewvalues():
            plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor="lightgreen", edgecolor="k"))

        for s in path[:-1]:
            (y, x) = np.unravel_index(s % self.base_s, self.map.shape)
            plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.2, facecolor="red", edgecolor="k"))

        plt.ylim(0, self.map.shape[0])
        plt.xlim(0, self.map.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

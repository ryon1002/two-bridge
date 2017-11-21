import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import squareform, pdist
import CollaboratePredictabilityWorld


class ItemPickWorld(CollaboratePredictabilityWorld.CollaboratePredictabilityWorld):
    def __init__(self):
        self.map = np.zeros((8, 15))
        self.items = [(1, 2), (6, 1), (2, 6), (5, 7), (1, 12), (5, 13)]
        self.human = (1, 9)
        self.agent = (3, 9)

        self.goal_ids = np.arange(len(self.items))
        self.dists = squareform(pdist(np.array(self.items + [self.human] + [self.agent]), metric="cityblock"))

    def make_pre_cond(self, pre_assign):
        return tuple([(pre[-1], self.single_orderd_assign_cost(pre, (self.goal_ids[-1] + i + 1, 0)))
                       if len(pre) > 0 else (self.goal_ids[-1] + i + 1, 0) for i, pre in enumerate(pre_assign)])

    def single_orderd_assign_cost(self, o_assign, pre_cond):
        assign = (pre_cond[0],) + o_assign
        return sum([self.dists[assign[i], assign[i + 1]] for i in range(len(assign) - 1)]) + pre_cond[1]

    def show_world(self, path=()):
        for y, x in itertools.product(*[range(i) for i in self.map.shape]):
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))

        for (y, x) in self.items:
            plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor="lightblue", edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.human[1] + 0.5, self.human[0] + 0.5),
                                           0.4, facecolor="lightgreen", edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.agent[1] + 0.5, self.agent[0] + 0.5),
                                           0.4, facecolor="pink", edgecolor="k"))
        plt.ylim(0, self.map.shape[0])
        plt.xlim(0, self.map.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
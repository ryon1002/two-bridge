import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import world.GridWorldMDPBase


class OneItemPickWorld(world.GridWorldMDPBase.GridWorldMDPBase):
    def __init__(self, shape, item):
        self.shape = shape
        self.item = item
        super(OneItemPickWorld, self).__init__(shape[1], shape[0], 1, 0, d=1)
        self.r[:, :, :] = -0.01
        self.set_goal(np.ravel_multi_index(self.item, shape), 1)

    def set_goal(self, g, reward):
        print g
        self.set_terminate_states(g)
        self.r[:, :, g] = reward
        self.r[:, g, g] = 0

    def show_world(self, path=()):
        for y, x in itertools.product(*[range(i) for i in self.shape]):
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))

        plt.gca().add_patch(patches.Circle((self.item[1] + 0.5, self.item[0] + 0.5), 0.4, facecolor="lightgreen", edgecolor="k"))

        for s in path[:-1]:
            (y, x) = np.unravel_index(s % self.base_s, self.map.shape)
            plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.2, facecolor="red", edgecolor="k"))

        plt.ylim(0, self.shape[0])
        plt.xlim(0, self.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

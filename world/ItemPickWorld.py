import numpy as np
import GridWorldMDPBase
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class ItemPickWorld(GridWorldMDPBase.GridWorldMDPBase):
    def __init__(self, world_map, item_map):
        self.map = world_map
        self.item = item_map
        super(ItemPickWorld, self).__init__(self.map.shape[1], self.map.shape[0], 1, 0, d=0.999)

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
        color_map = {0: "w", 1: "cyan", 2: "pink"}
        for s in range(self.s):
            (y, x) = np.unravel_index(s, self.map.shape)
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color_map[self.map[y, x]], edgecolor="k"))
            if self.item[y, x] == 1:
                plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor="gold", edgecolor="k"))

        for s in path:
            (y, x) = np.unravel_index(s, self.map.shape)
            plt.gca().add_patch(patches.Circle((x + 0.5, y + 0.5), 0.2, facecolor="red", edgecolor="k"))

        plt.ylim(0, self.map.shape[0])
        plt.xlim(0, self.map.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

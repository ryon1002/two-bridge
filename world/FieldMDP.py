import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import GridWorldMDPBase

class FieldMDP(GridWorldMDPBase.GridWorldMDPBase):
    def __init__(self, worldmap, **kwargs):
        self.keyaMap = {"up":0, "down":3, "right":2, "left":1}
        self.map = worldmap
        GridWorldMDPBase.GridWorldMDPBase.__init__(self, self.map.shape[1], self.map.shape[0], 1, 0, **kwargs)
        self.fmap = self.map.flatten()[::-1]
        
        for a, s in itertools.product(range(4), range(self.s)):
            if self.fmap[np.argmax(self.t[a, s, :])] == 1:
                self.t[a, s, :] = 0
                self.t[a, s, s] = 1
        self.r[:4, :, :] = -0.01

    def setAgent(self, s):
        self.agentState = s

    def setSidekick(self, s):
        self.sidekickState = s

    def setGoal(self, g):
        self.setTerminateStates(g)
        self.r[:, :, g] = 1
        self.r[:, g, g] = 0

    def makeRoad(self, s, reward=-0.01):
        pos = np.array(np.unravel_index(s, self.shape))
        for offset in np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]):
            s2 = np.ravel_multi_index(pos + offset, self.shape)
            if self.fmap[s2] == 0: self.connect(s, s2, reward)

    def connect(self, s1, s2, reward):
        s1y, s1x = np.unravel_index(s1, self.shape)
        s2y, s2x = np.unravel_index(s2, self.shape)
        if abs(s1y - s2y) == 1 and s1x == s2x:
            a12 , a21 = (3, 0) if s1y - s2y == 1 else (0, 3)
        elif abs(s1x - s2x) == 1 and s1y == s2y:
            a12 , a21 = (1, 2) if s1y - s2y == 1 else (2, 1)
        self.t[a12, s1, s1] = 0
        self.t[a12, s1, s2] = 1
        self.r[a12, s1, s2] = reward
        self.t[a21, s2, s2] = 0
        self.t[a21, s2, s1] = 1
        self.r[a21, s2, s1] = reward

    def printMap(self):
        sMap = np.zeros_like(self.map, dtype=object)
        for s in range(self.s):
            y, x = np.unravel_index(s, self.shape)
            sMap[self.shape[0] - y - 1, x] = (s, int(self.map[self.shape[0] - y - 1, x]))
        print sMap

    def showWorld(self):
        for s in range(self.s):
            (y, x) = np.unravel_index(s, self.shape)
            color = "cyan" if self.fmap[s] == 1 else "w"
            if s in [43, 47] :  color = "lightcyan"
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color))
        plt.ylim(0, self.shape[0])
        plt.xlim(0, self.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().add_patch(patches.Circle(np.unravel_index(self.agentState % 81, self.shape)[::-1] + np.array([0.5, 0.5]), radius=0.1, facecolor="red"))        
#         plt.gca().add_patch(patches.Circle(np.unravel_index(self.sidekickState % 81, self.shape)[::-1] + np.array([0.5, 0.5]), radius=0.1, facecolor="blue"))        
        y, x = np.unravel_index(self.sidekickState % 81, self.shape)
        plt.gca().add_patch(patches.Rectangle((x + 0.1, y + 0.1), 0.8, 0.8, facecolor="g"))

    def move(self, event):
        if event.key in self.keyaMap:
            prevState = self.agentState
            self.agentState = np.argmax(self.t[self.keyaMap[event.key], self.agentState])
            print self.agentState
            return prevState, self.keyaMap[event.key]
        return None

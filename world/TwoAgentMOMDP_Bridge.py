import numpy as np
from core import TwoAgentMOMDP

class TwoAgentMOMDP_Bridge(TwoAgentMOMDP.TwoAgentMOMDP):
    def __init__(self, agents, sidekick, bridge, goal): # depth=1, bs=None):
        self.mainFirst = True
        self.makeTransition(sidekick, agents)
        self.addSinkState(None, agents[0].shrinkMap[goal])
        
        self.ty = np.zeros((self.y, self.a, self.x, self.y))
        for y in range(self.y):
            for x in range(self.x):
                for a in range(self.a):
                    self.ty[y, a, x, y] = 1
        
        self.r = np.zeros((self.a, self.x, self.y))
        goal = agents[0].shrinkMap[goal]
        self.r[:4, :, :] = -0.01
        self.r[:, -1, :] = 0
        for a in range(self.a):
            self.r[a, self.getState(None, goal)] = 1
            for b in bridge:
                a_b = agents[0].shrinkMap[b]
                self.r[a, self.getState(None, a_b), :] = -0.4
                self.r[a, self.getState(sidekick.shrinkMap[b], a_b), :] = -0.01
            
        self.preCulc()

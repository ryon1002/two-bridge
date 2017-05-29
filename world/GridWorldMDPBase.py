import numpy as np
from core import MDP

class GridWorldMDPBase(MDP.MDP):
    dirActionIdx = {"^":np.array([1, 0]), "<":np.array([0, -1]), ">":np.array([0, 1]), "v":np.array([-1, 0]), "o":np.array([0, 0])}
    
    def __init__(self, xNum, yNum, prob=1.0, errProb=0.0, **kwargs):
        self.aMap = {0:"^", 1:"<", 2:">", 3:"v", 4:"o"}
        self.shape = (yNum, xNum)
        MDP.MDP.__init__(self, yNum * xNum, len(self.aMap), **kwargs)
        a_t = np.ones((self.a, self.a)) * errProb
        np.fill_diagonal(a_t, prob)
        for s in range(self.s):
            nexts = [self.getNextStateForDirAction(s, self.aMap[a]) for a in range(self.a)]
            for i, ns in enumerate(nexts):
                self.t[:, s, ns] += a_t[:, i]
        
    def getNextStateForDirAction(self, s, dirAction):
        nextPos = np.unravel_index(s, self.shape) + GridWorldMDPBase.dirActionIdx[dirAction]
        return np.ravel_multi_index(nextPos, self.shape, mode="clip")

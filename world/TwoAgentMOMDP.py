import numpy as np
import itertools
import FieldMDP2

class TwoAgentMOMDP(object):
    def __init__(self, agents, sidekick, bridge, goal): # depth=1, bs=None):
        self.y = len(agents)
        self.x = agents[0].s * sidekick.s + 1
        self.a = sidekick.a
        self.t = np.zeros((self.y, self.a, self.x, self.x))
        self.r = np.zeros((self.a, self.x, self.y))
        
        self.beta = 50
        for y, agent in enumerate(agents):
            agent.setGoal(agent.shrinkMap[goal])
            agent.doValueItaration()
            aProb = self.softmax(agent.q)
            
            for a_s in range(agent.s):
                t = np.sum(agent.t[:, a_s, :] * aProb[:, a_s][:, np.newaxis], axis=0)
                for t_i in np.where(t > 0)[0]:
                    self.t[y, :, a_s * sidekick.s:(a_s + 1) * sidekick.s, t_i * sidekick.s:(t_i + 1) * sidekick.s] = sidekick.t * t[t_i]
            g = agent.shrinkMap[goal]
            self.t[y, :, g * sidekick.s:(g + 1) * sidekick.s, 0:-1] = 0
            self.t[y, :, g * sidekick.s:(g + 1) * sidekick.s, -1] = 1
            self.t[y, :, -1, -1] = 1
        
        goal = agents[0].shrinkMap[goal]
        self.r[:4, :, :] = -0.01
        self.r[:, goal * sidekick.s:(goal + 1) * sidekick.s, :] = 1
        self.r[:, -1, :] = 0
        for b in bridge:
            a_b = agents[0].shrinkMap[b]
            self.r[:, a_b * sidekick.s:(a_b + 1) * sidekick.s, :] = -0.4
            self.r[:, a_b * sidekick.s + sidekick.shrinkMap[b], :] = -0.01
            
        # For calc Avector
        self.valid_nxs = np.sum(self.t, axis=0) > 0
        self.valid_nxs = {a:{x:np.where(self.valid_nxs[a][x])[0] for x in range(self.valid_nxs.shape[1])} for a in range(self.valid_nxs.shape[0])}

    def softmax(self, arr):
        ret = np.exp(arr * self.beta)
        return ret / np.sum(ret, axis=0)

    def calcAvector(self, d=1, bs=None, withA=True):
        if d == 1:
            self.aVector = {x:self.r[:, x, :].copy() for x in range(self.x)}
            return
        self.calcAvector(d - 1, bs, False)
        print d
        aVector = {}
        for x in range(self.x):
            aVector[x] = {}
            for a in range(self.a):
                nxs = self.t[:, a, x]
                aiList = [nxs[:, nx] * self.aVector[nx] for nx in self.valid_nxs[a][x]]
                aiLenList = [len(aiList[i]) for i in range(len(aiList))]
                
                aiList2 = np.zeros((np.prod(aiLenList), self.y))
                for m, i in enumerate(itertools.product(*[range(l) for l in aiLenList])):
                    aiList2[m] = np.sum([aiList[n][j] for n, j in enumerate(i)], axis=0)
                aiList2 = self.uniqueFowRaw(aiList2)
                aVector[x][a] = self.r[a, x] + aiList2
        if withA:
            self.aVectorA = {x:{a:self.prune(vector, bs) for a, vector in vectorA.viewitems()} for x, vectorA in aVector.viewitems()} if bs is not None else aVector
        else :
            self.aVector = {x:self.prune(np.concatenate(vector.values(), axis=0), bs) for x, vector in aVector.viewitems()} if bs is not None else aVector

    def uniqueFowRaw(self, a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
    
    def prune(self, aVector, bs):
        index = np.unique(np.argmax(np.dot(aVector, bs.T), axis=0))
        return aVector[index]

    def valueA(self, x, a, b):
        return np.max(np.dot(self.aVectorA[x][a], b))

    def getBestAction(self, x, b):
        print b
        valueMap = {k:np.max(np.dot(v, b))for k, v in self.aVectorA[x].viewitems()}
        print sorted(valueMap.viewitems(), key=lambda x:x[1])
        print sorted(valueMap.viewitems(), key=lambda x:x[1])[-1][0]
        return sorted(valueMap.viewitems(), key=lambda x:x[1])[-1][0]
#         return 4

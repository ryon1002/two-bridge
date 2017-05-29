import numpy as np
import itertools

class MOMDP(object):    
    def preCulc(self):
        # For calc Avector
        self.valid_nxs = np.sum(self.t, axis=0) > 0
        self.valid_nxs = {a:{x:np.where(self.valid_nxs[a][x])[0] for x in range(self.valid_nxs.shape[1])} for a in range(self.valid_nxs.shape[0])}

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
        valueMap = {k:np.max(np.dot(v, b))for k, v in self.aVectorA[x].viewitems()}
        return sorted(valueMap.viewitems(), key=lambda x:x[1])[-1][0]

import numpy as np
import itertools

class Tiger(object):
    def __init__(self, depth=1, bs=None):
#         self.x = 6
        self.x = 6
        self.y = 2
        self.a = 3
        self.t = np.zeros((self.y, self.a, self.x, self.x))
        self.r = np.zeros((self.a, self.x, self.y))
        
        for x in range(self.x - 1):
            self.t[0, -1, x, min(x + 1, self.x - 2)] = 0.85
            self.t[0, -1, x, max(0, x - 1)] = 0.15
            self.t[1, -1, x, min(x + 1, self.x - 2)] = 0.15
            self.t[1, -1, x, max(0, x - 1)] = 0.85
        self.t[0, :-1, :, -1] = 1
        self.t[1, :-1, :, -1] = 1
        self.t[:, :, -1, -1] = 1

        self.r[0, :-1, 0] = 10
        self.r[1, :-1, 0] = -100
        self.r[0, :-1, 1] = -100
        self.r[1, :-1, 1] = 10
        self.r[2, :-1, :] = -1

        self.calcAvector(depth, bs)

    def calcAvector(self, d=1, bs=None):
        if d == 1:
            self.aVector = {x:self.r[:, x, :].copy() for x in range(self.x)}
            return
        self.calcAvector(d - 1, bs)
        aVector = {}
        for x in range(self.x):
            aVector[x] = np.zeros((0, self.y))
            for a in range(self.a):
                aiList = []
                aiLenList = []
                nxs = self.t[:, a, x]
                valid_nxs = np.where(np.sum(nxs, axis=0) > 0)[0]
                for nx in valid_nxs:
                    aiList.append(nxs[:, nx] * self.aVector[nx])
                    aiLenList.append(range(len(self.aVector[nx])))
        
                aiList2 = np.zeros((0, self.y))
                for i in itertools.product(*aiLenList):
                    aiList2 = np.vstack((aiList2, np.sum([aiList[n][j] for n, j in enumerate(i)], axis=0)))
                aiList2 = self.uniqueFowRaw(aiList2)
                aVector[x] = np.vstack((aVector[x], self.r[a, x] + aiList2))
        self.aVector = {x:self.prune(vector, bs) for x, vector in aVector.viewitems()} if bs is not None else aVector
                    
    def uniqueFowRaw(self, a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
    
    def value(self, x, b):
        return np.max(np.dot(self.aVector[x], b))

    def prune(self, aVector, bs):
        index = np.unique(np.argmax(np.dot(aVector, bs.T), axis=0))
        return aVector[index]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    
    b = np.concatenate(([b1], [b2]), axis=0).T

    env = Tiger(1, b)
    v = np.array([env.value(1, b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)
    
    env2 = Tiger(10, b)
    v = np.array([env2.value(1, b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)

    env3 = Tiger(20, b)
    v = np.array([env3.value(1, b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)
# 
#     import datetime
#     start = datetime.datetime.now()
#     env4 = Tiger(30, b)
#     v = np.array([env4.value(b[i]) for i in range(len(b))])
#     print datetime.datetime.now() - start
#     plt.plot(b[:, 0], v)

    plt.show()
    

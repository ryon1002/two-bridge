import numpy as np
import itertools

class Tiger(object):
    def __init__(self, depth=1, bs=None):
        self.s = 3
        self.a = 3
        self.o = 3
        self.t = np.zeros((self.a, self.s, self.s))
        self.r = np.zeros((self.a, self.s))
        self.ot = np.zeros((self.a, self.s, self.o))
        
        self.t[:2, :, -1] = 1
        self.t[-1] = np.identity(3)
        
        self.r[:] = -100
        for i in range(2):
            self.r[i, i] = 10
        self.r[-1, :] = -1
        self.r[:, -1] = 0
        
        self.ot[:, -1, -1] = 1
        for i in range(2):
            self.ot[:, i, i] = 0.85
            self.ot[:, i, 1 - i] = 0.15
        self.orange = np.arange(self.o)
        self.calcAvector(depth, bs)

    def calcAvector(self, d=1, bs=None):
        if d == 1:
            self.aVector = self.r.copy()
            return
        self.calcAvector(d - 1, bs)
        aVector = np.zeros((0, self.s))
        for a in range(self.a):
            nb = np.dot(self.t[a], self.ot[a])
            aiList = np.zeros((self.o, self.aVector.shape[0], self.aVector.shape[1]))
            for o in range(self.o):
                aiList[o] = nb[o] * self.aVector            
            
            aiList2 = np.zeros((len(self.aVector) ** self.o, self.aVector.shape[1]))
            for m, i in enumerate(itertools.product(range(len(self.aVector)), repeat=self.o)):
                aiList2[m] = np.sum(aiList[self.orange, i], axis=0)
#                 aiList2[m] = np.sum([aiList[n, j] for n, j in enumerate(i)], axis=0)
            aiList2 = self.uniqueFowRaw(aiList2)
            aVector = np.vstack((aVector, self.r[a] + aiList2))
        self.aVector = self.prune(aVector, bs) if bs is not None else aVector
    
    def uniqueFowRaw(self, a):
        return np.unique(a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))).view(a.dtype).reshape(-1, a.shape[1])
    
    def value(self, b):
        return np.max(np.dot(self.aVector, b))

    def prune(self, aVector, bs):
        index = np.unique(np.argmax(np.dot(aVector, bs.T), axis=0))
        return aVector[index]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    
    b = np.concatenate(([b1], [b2], [np.zeros((len(b1)))]), axis=0).T

    env = Tiger(1, b)
    v = np.array([env.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)


    env2 = Tiger(3, b)
    v = np.array([env2.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)
 
    env3 = Tiger(6, b)
    v = np.array([env3.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)
 
    import datetime
    start = datetime.datetime.now()
    env4 = Tiger(30, b)
    v = np.array([env4.value(b[i]) for i in range(len(b))])
    print datetime.datetime.now() - start
    plt.plot(b[:, 0], v)

    plt.show()
    

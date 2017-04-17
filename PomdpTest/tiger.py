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
        self.calcAvector(depth, bs)

    def calcAvector(self, d=1, bs=None):
        if d == 1:
            self.aVector = self.r.copy()
            return
        self.calcAvector(d - 1, bs)
        aVector = np.zeros((0, self.s))
        for a in range(self.a):
            nb = np.dot(self.t[a], self.ot[a])
            aiList = np.zeros((0, self.aVector.shape[0], self.aVector.shape[1]))
            for o in range(self.o):
                aiList = np.vstack((aiList, [nb[o] * self.aVector]))
            aiList2 = np.zeros((0, self.aVector.shape[1]))
            for i in itertools.product(range(len(self.aVector)), repeat=self.o):
                aiList2 = np.vstack((aiList2, np.sum([aiList[n, j] for n, j in enumerate(i)], axis=0)))
            aiList2 = np.vstack({tuple(row) for row in aiList2})
            aVector = np.vstack((aVector, self.r[a] + aiList2))
        self.aVector = self.prune(aVector, bs) if bs is not None else aVector
        print self.aVector.shape
    
    def value(self, b):
        return np.max(np.dot(self.aVector, b))

    def prune(self, aVector, bs):
        index = list({np.argmax(np.dot(aVector, b)) for b in bs})
        return aVector[index]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    b1 = np.arange(0, 1.01, 0.01)
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

    env4 = Tiger(10, b)
    v = np.array([env4.value(b[i]) for i in range(len(b))])
    plt.plot(b[:, 0], v)

    plt.show()
    

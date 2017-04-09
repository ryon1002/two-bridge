import numpy as np

class Mdp(object):
    def __init__(self, s, a, d=0.99, inits=[0], **kwargs):
        self.s = s
        self.inits = inits
        self.a = a
        self.d = d
        self.t = np.zeros((self.a, self.s, self.s))
        self.r = np.zeros((self.a, self.s, self.s))
        self.q = np.zeros((self.s, self.a))
        self.terminateStates = None

    def getNextState(self, action, state):
        return np.random.choice(np.arange(len(self.t[0])), p=self.t[action, state])

    def doValueItaration(self):
        v = np.zeros(self.t.shape[1])
        for _i in range(500):
            n_v = np.sum(self.t * self.r + self.t * v * self.d, axis=2)
            n_v = np.max(n_v, axis=0)
            chk = np.sum(np.abs(n_v - v))
            if chk < 1e-6:
                break
            v = n_v
        self.q = np.sum(self.t * self.r + self.t * n_v * self.d, axis=2)

    def isTerminate(self, s):
        if self.terminateStates is not None and s in self.terminateStates:
            return True
        return False

    def setTerminateStates(self, s):
        self.terminateStates = s
        self.t[:, s, :] = 0
        self.t[:, s, s] = 1

    def pickInitialState(self):
        return np.random.choice(self.inits)


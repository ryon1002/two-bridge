import numpy as np
import MOMDP

class TwoAgentMOMDP(MOMDP.MOMDP):
    def makeTransition(self, main, targets):
        self.y = len(targets)
        self.x = main.s * targets[0].s
        self.a = main.a
        self.tx = np.zeros((self.y, self.a, self.x, self.x))
        self.ty = np.zeros((self.y, self.a, self.x, self.y))
        self.main_s = main.s
        self.target_s = targets[0].s

        for y, target in enumerate(targets):
            for t_s in range(target.s):
                t = np.sum(target.t[:, t_s, :] * target.aProb[:, t_s][:, np.newaxis], axis=0)
                for t_i in np.where(t > 0)[0]:
                    for m_s in range(main.s):
                        for a in range(self.a):
                            self.tx[y, a, self.getState(m_s, t_s), self.getState(None, t_i)] = main.t[a, m_s] * t[t_i]
    
    def addSinkState(self, mainState, targetState):
        self.x = self.x + 1
        tmptx = self.tx
        self.tx = np.zeros((self.y, self.a, self.x, self.x))
        self.tx[:, :, :-1, :-1] = tmptx
        for y in range(self.y):
            for a in range(self.a):
                self.tx[y, a, self.getState(mainState, targetState), 0:-1] = 0
                self.tx[y, a, self.getState(mainState, targetState), -1] = 1
            self.tx[y, :, -1, -1] = 1
    
    def getState(self, mainState, targetState):
        if self.mainFirst:
            if mainState is None:
                return np.arange(self.main_s) + targetState * self.main_s
            elif targetState is None:
                return mainState + self.main_s * np.arange(self.target_s)
            return mainState + targetState * self.main_s
        else:
            if targetState is None:
                return np.arange(self.targetState) + mainState * self.target_s
            elif mainState is None:
                return targetState + self.target_s * np.arange(self.main_s)
            return targetState + mainState * self.target_s
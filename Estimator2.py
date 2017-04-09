import numpy as np
from world import FieldMDP
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, worldmap):
        goalList = [43, 47]

        self.beta = 10
        pList = []
        for g in goalList:
            field = FieldMDP.FieldMDP(1 - worldmap)
            field.setGoal(g)
            field.doValueItaration()
            q = self.softmax(field.q)
            pList.append(q)
        
        pList = np.array(pList)
        self.pList = pList / np.sum(pList, axis=0)
        self.belief = np.ones((len(pList))) / len(pList) 
        self.belief = np.array([0.5, 0.5])
        
    def softmax(self, arr):
        ret = np.exp(arr * self.beta)
        return ret / np.sum(ret, axis=0)

    def updateBelief(self, s, a):
        self.belief *= self.pList[:, a, s]
        self.belief /= np.sum(self.belief)
    
    def showBelief(self):
        plt.ylim(0, 1)
        plt.bar(np.arange(len(self.belief)), self.belief, color=["r", "b"])

import numpy as np
from world import FieldMDP
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, worldmap):
        roadList = [43, 47]

        self.beta = 10
        pList = []
        qList = []
        for r in range(len(roadList)):
            field = FieldMDP.FieldMDP(worldmap)
            field.setGoal(80)
            field.makeRoad(roadList[r], -0.01)
            field.makeRoad(roadList[1 - r], -0.1)
            field.doValueItaration()
            qList.append(field.q)
            q = self.softmax(field.q)
            pList.append(q)
        
        pList = np.array(pList)
        self.qList = np.array(qList)
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

import numpy as np
from world import FieldMDP
import matplotlib.pyplot as plt

class Sidekick():
    def __init__(self, world, worldmap, estimator, estimator2):
        self.world = world
        goalList = [43, 47]
        self.qList = []
        self.baseMdp = FieldMDP.FieldMDP(1 - worldmap)
        
        for g in goalList:
            field = FieldMDP.FieldMDP(1 - worldmap)
            field.setGoal(g)
            field.doValueItaration()
            self.qList.append(field.q)
        self.qList = np.array(self.qList)
        self.estimator = estimator
        self.estimator2 = estimator2

        roadList = [43, 47]
        self.beta = 10
#         pList = []
        qList = []
        for r in range(len(roadList)):
            field = FieldMDP.FieldMDP(worldmap)
            field.setGoal(80)
            field.makeRoad(64, -0.035)
            field.makeRoad(roadList[r], -0.01)
            field.makeRoad(roadList[1 - r], -0.1)
            field.doValueItaration()
            qList.append(field.q)

        self.gtqList = np.array(qList)
        self.toAgentBelief = np.array([0.5, 0.5])

    def doBestAction(self):
        prevState = self.world.sidekickState
        q = np.sum(self.estimator.belief[i] * self.qList[i] for i in range(2))
        action = np.argmax(q[:, self.world.sidekickState])
        self.world.sidekickState = np.argmax(self.baseMdp.t[action, self.world.sidekickState])
        return prevState, action

    def doBestAction2(self):
        prevState = self.world.sidekickState
        
        v = np.max(self.gtqList[:, :, self.world.agentState], axis=1)
        nextB = self.estimator2.pList[:, :, self.world.sidekickState] * self.estimator2.belief[:, np.newaxis]
        nextB /= np.sum(nextB, axis=0)
        beta = 0.5

        toAgentBelief = beta * nextB + (1 - beta) * self.estimator.belief[:, np.newaxis]
        
        action = np.argmax(np.dot(v, nextB))
        self.toAgentBelief = toAgentBelief[:, action]
        
        self.world.sidekickState = np.argmax(self.baseMdp.t[action, self.world.sidekickState])
        return prevState, action
        
    def showBelief(self):
        plt.ylim(0, 1)
        plt.bar(np.arange(len(self.toAgentBelief)), self.toAgentBelief , color=["r", "b"])

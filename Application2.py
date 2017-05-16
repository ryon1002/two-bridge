import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from world import FieldMDP2, TwoAgentMOMDP
import numpy as np

class Application():
    def __init__(self, worldcsvfile):
        self.fig = plt.figure() 
        self.keyaMap = {"up":0, "down":3, "right":2, "left":1}
        self.worldcsv = np.loadtxt(worldcsvfile, delimiter=",")
        self.mapState = self.worldcsv.flatten()[::-1]
        
        self.bridge = [15, 19]
        self.agentState = 2
        self.sidekickState = 22
        self.goal = 32
        
        self.agents = []
        for i in range(len(self.bridge)):
            agent = FieldMDP2.FieldMDP2(self.worldcsv)
            for j in range(len(self.bridge)):
                if i == j :
                    agent.makeRoad(self.bridge[j])
                else:
                    agent.makeRoad(self.bridge[j], -0.2)        
            agent.shrinkState(self.agentState)
            self.agents.append(agent)

        self.sidekick = FieldMDP2.FieldMDP2(1 - self.worldcsv)
        for b in self.bridge:
            self.sidekick.setGoal(b)
        self.sidekick.shrinkState(self.sidekickState)
        
        self.collaborate = TwoAgentMOMDP.TwoAgentMOMDP(self.agents, self.sidekick, self.bridge, self.goal)
        self.collaborateState = self.agents[0].shrinkMap[self.agentState] * self.sidekick.s + self.sidekick.shrinkMap[self.sidekickState]
        b1 = np.arange(0, 1.01, 0.05)
#         b1 = np.arange(0, 1.01, 0.1)
        b2 = 1 - b1
        self.b = np.concatenate(([b1], [b2]), axis=0).T

        import pickle
        import datetime
        start = datetime.datetime.now()
#         self.collaborate.calcAvector(20, self.b, True)
        print datetime.datetime.now() - start
#         pickle.dump(self.collaborate.aVectorA, open('aVectorA.pkl', mode='w'))
        self.collaborate.aVectorA = pickle.load(open('aVectorA.pkl', mode='r'))
        
        self.belief = np.ones((len(self.bridge))) / len(self.bridge)  
        
        self.fig.canvas.mpl_connect('key_press_event', self.keyDown)
        self.draw()
        plt.show()
    
    def keyDown(self, event):
        sa = self.move(event)
        self.draw()

    def draw(self):
        plt.clf()
        G = gridspec.GridSpec(2, 2)
        self.fig.add_subplot(G[:, 0])
        self.showWorld()
        self.fig.add_subplot(G[0, 1])
        self.showBelief()
        self.fig.add_subplot(G[1, 1])
        self.showActionValue()
        self.fig.canvas.draw()

    def showWorld(self):
        for s in range(len(self.mapState)):
            (y, x) = np.unravel_index(s, self.worldcsv.shape)
            color = "cyan" if self.mapState[s] == 1 else "w"
            if s in self.bridge :  color = "lightcyan"
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color))
        plt.ylim(0, self.worldcsv.shape[0])
        plt.xlim(0, self.worldcsv.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().add_patch(patches.Rectangle(np.unravel_index(self.sidekickState, self.worldcsv.shape)[::-1] + np.array([0.1, 0.1]), 0.8, 0.8, facecolor="g"))        
        plt.gca().add_patch(patches.Circle(np.unravel_index(self.agentState, self.worldcsv.shape)[::-1] + np.array([0.5, 0.5]), radius=0.1, facecolor="red"))        

    def showActionValue(self):
        leg = []
        max = -1
        min = 1
        for a in range(self.collaborate.a):
            v = np.array([self.collaborate.valueA(self.collaborateState, a, self.b[i]) for i in range(len(self.b))])
            max = np.max([max, np.max(v)])
            min = np.min([min, np.min(v)])
            leg.append(plt.plot(self.b[:, 0], v)[0])
        plt.legend(leg, ["^", "<", ">", "v", "o"])
        plt.plot([self.belief[0], self.belief[0]], [max, min])
        plt.show()

    def showBelief(self):
        plt.ylim(0, 1)
        plt.bar(np.arange(len(self.belief)), self.belief, color="orange")

    def updateBelief(self, a, s, n_s):
        self.belief *= self.collaborate.t[:, a, s, n_s]
        self.belief /= np.sum(self.belief)

    def move(self, event):
        if event.key in self.keyaMap:
            prevState = self.collaborateState
            a_s = self.agents[0].getNextState(self.keyaMap[event.key], self.agents[0].shrinkMap[self.agentState])
            self.agentState = self.agents[0].i_shrinkMap[a_s]
            s_a = self.collaborate.getBestAction(self.collaborateState, self.belief)
            s_s = self.sidekick.getNextState(s_a, self.sidekick.shrinkMap[self.sidekickState])
            self.sidekickState = self.sidekick.i_shrinkMap[s_s]
            self.collaborateState = self.agents[0].shrinkMap[self.agentState] * self.sidekick.s + self.sidekick.shrinkMap[self.sidekickState]
            self.updateBelief(s_a, prevState, self.collaborateState)

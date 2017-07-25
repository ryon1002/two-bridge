import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from world import FieldMDP, TwoAgentMOMDPBridge
import numpy as np


class Application(object):
    def __init__(self, worldcsvfile):
        self.fig = plt.figure()
        self.keyaMap = {"up": 0, "down": 3, "right": 2, "left": 1}
        self.worldcsv = np.loadtxt(worldcsvfile, delimiter=",")
        self.mapState = self.worldcsv.flatten()[::-1]

        self.bridge = [15, 19]
        self.agentState = 2
        self.sidekickState = 22
        self.goal = 32
        self.beta = 50

        self.agents = []
        for i in range(len(self.bridge)):
            agent = FieldMDP.FieldMDP(self.worldcsv)
            for j in range(len(self.bridge)):
                if i == j:
                    agent.make_road(self.bridge[j])
                else:
                    agent.make_road(self.bridge[j], -0.2)
            agent.shrink_state(self.agentState)
            self.agents.append(agent)
            agent.set_goal(agent.shrinkMap[self.goal])
            agent.do_value_iteration()
            agent.aProb = agent.softmax(agent.q, self.beta)

        self.sidekick = FieldMDP.FieldMDP(1 - self.worldcsv)
        for b in self.bridge:
            self.sidekick.set_goal(b)
        self.sidekick.shrink_state(self.sidekickState)

        self.agentState = self.agents[0].shrinkMap[self.agentState]
        self.sidekickState = self.sidekick.shrinkMap[self.sidekickState]

        self.collaborate = TwoAgentMOMDPBridge.TwoAgentMOMDPBridge(self.agents, self.sidekick, self.bridge, self.goal)
        self.collaborateState = self.collaborate.baseMDP.get_state(self.sidekickState, self.agentState)
        b1 = np.arange(0, 1.01, 0.05)
        #         b1 = np.arange(0, 1.01, 0.1)
        b2 = 1 - b1
        self.b = np.concatenate(([b1], [b2]), axis=0).T

        start = datetime.datetime.now()
        self.collaborate.calc_a_vector(15, self.b, True)
        # check_vector = pickle.load(open('aVectorA.pkl', mode='r'))
        # for s, aq in self.collaborate.a_vector_a.viewitems():
        #     for a, q in aq.viewitems():
        #         if not np.array_equal(q, check_vector[s][a]):
        #             #             if not np.array_equal(self.collaborate.aVectorA[k], self.collaborate.aVectorA[k]):
        #             print s, a, self.collaborate.a_vector_a[s][a]
        #             print s, a, check_vector[s][a]
        #             exit()
        # print "true"
        # exit()
        #         pickle.dump(self.collaborate.aVectorA, open('aVectorA.pkl', mode='w'))
        print datetime.datetime.now() - start
        # self.collaborate.a_vector_a = pickle.load(open('aVectorA.pkl', mode='r'))

        self.belief = np.ones((len(self.bridge))) / len(self.bridge)

        self.fig.canvas.mpl_connect('key_press_event', self.key_down)
        self.draw()
        plt.show()

    def key_down(self, event):
        self.move(event)
        self.draw()

    def draw(self):
        plt.clf()
        grid = gridspec.GridSpec(2, 2)
        self.fig.add_subplot(grid[:, 0])
        self.show_world()
        self.fig.add_subplot(grid[0, 1])
        self.show_belief()
        self.fig.add_subplot(grid[1, 1])
        self.show_action_value()
        self.fig.canvas.draw()

    def show_world(self):
        for s in range(len(self.mapState)):
            (y, x) = np.unravel_index(s, self.worldcsv.shape)
            color = "cyan" if self.mapState[s] == 1 else "w"
            if s in self.bridge:
                color = "lightcyan"
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color))
        plt.ylim(0, self.worldcsv.shape[0])
        plt.xlim(0, self.worldcsv.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().add_patch(patches.Rectangle(
            np.unravel_index(self.sidekick.i_shrinkMap[self.sidekickState], self.worldcsv.shape)[::-1] + np.array(
                [0.1, 0.1]), 0.8, 0.8, facecolor="g"))
        plt.gca().add_patch(patches.Circle(
            np.unravel_index(self.agents[0].i_shrinkMap[self.agentState], self.worldcsv.shape)[::-1] + np.array(
                [0.5, 0.5]), radius=0.1, facecolor="red"))

    def show_action_value(self):
        leg = []
        v_max = -1
        v_min = 1
        for a in range(self.collaborate.a):
            v = np.array([self.collaborate.value_a(self.collaborateState, a, self.b[i]) for i in range(len(self.b))])
            v_max = np.max([v_max, np.max(v)])
            v_min = np.min([v_min, np.min(v)])
            leg.append(plt.plot(self.b[:, 0], v)[0])
        plt.legend(leg, ["^", "<", ">", "v", "o"])
        plt.plot([self.belief[0], self.belief[0]], [v_max, v_min])
        plt.show()

    def show_belief(self):
        plt.ylim(0, 1)
        plt.bar(np.arange(len(self.belief)), self.belief, color="orange")

    def update_belief(self, a, s, n_s):
        self.belief *= self.collaborate.tx[:, a, s, n_s]
        print self.belief
        print self.collaborate.tx[:, a, s, n_s]
        self.belief /= np.sum(self.belief)

    def move(self, event):
        if event.key in self.keyaMap:
            prev_state = self.collaborateState
            self.agentState = self.agents[0].get_next_state(self.keyaMap[event.key], self.agentState)
            s_a = self.collaborate.get_best_action(self.collaborateState, self.belief)
            self.sidekickState = self.sidekick.get_next_state(s_a, self.sidekickState)
            self.collaborateState = self.collaborate.baseMDP.get_state(self.sidekickState, self.agentState)
            self.update_belief(s_a, prev_state, self.collaborateState)

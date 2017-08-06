import pickle
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from world import TwoAgentMOMDPBridge
from solver import SarsopMomdp
import numpy as np


class Application(object):
    def __init__(self, world_csv_file):
        self.fig = plt.figure()
        self.key_a_map = {"up": 0, "down": 3, "right": 2, "left": 1}
        self.world_csv = np.loadtxt(world_csv_file, delimiter=",")
        self.map_state = self.world_csv.flatten()[::-1]

        bridge = [15, 19]
        agent_pos = 2
        sidekick_pos = 22
        goal = 32
        beta = 50

        self.collaborate = TwoAgentMOMDPBridge.TwoAgentMOMDPBridge(self.world_csv, agent_pos, sidekick_pos,
                                                                   bridge, goal, beta)
        self.collaborate_state = self.collaborate.get_state(sidekick_pos, agent_pos)

        # solver = SarsopMomdp.SarsopMOMDP()
        # self.collaborate.a_vector_a = solver.solve(self.collaborate, self.collaborateState)

        b1 = np.arange(0, 1.01, 0.05)
        # b1 = np.arange(0, 1.01, 0.1)
        self.b = np.concatenate(([b1], [1 - b1]), axis=0).T

        # start = datetime.datetime.now()
        # self.collaborate.calc_a_vector(10, self.b, True)
        # check_vector = pickle.load(open('aVector.pkl', mode='r'))
        # # print check_vector[0]
        # for s, aq in self.collaborate.a_vector_a.viewitems():
        #     for a, q in aq.viewitems():
        #         if not np.array_equal(q, check_vector[s][a]):
        #             #             if not np.array_equal(self.collaborate.aVectorA[k], self.collaborate.aVectorA[k]):
        #             print s, a, self.collaborate.a_vector_a[s][a]
        #             print s, a, check_vector[s][a]
        #             exit()
        # print "true"
        # exit()
        # print datetime.datetime.now() - start
        # # pickle.dump(self.collaborate.a_vector_a, open('aVector.pkl', mode='w'))
        self.collaborate.a_vector_a = pickle.load(open('aVector.pkl', mode='r'))

        self.belief = np.ones(self.collaborate.y) / self.collaborate.y

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
        for s in range(len(self.map_state)):
            (y, x) = np.unravel_index(s, self.world_csv.shape)
            color = "cyan" if self.map_state[s] == 1 else "w"
            if s in self.collaborate.bridge:
                color = "lightcyan"
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor=color))
        plt.ylim(0, self.world_csv.shape[0])
        plt.xlim(0, self.world_csv.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        sidekick_pos, agent_pos = self.collaborate.get_pos(self.collaborate_state)
        plt.gca().add_patch(patches.Rectangle(
            np.unravel_index(sidekick_pos, self.world_csv.shape)[::-1] + np.array( [0.1, 0.1]), 0.8, 0.8,
            facecolor="g"))
        plt.gca().add_patch(patches.Circle(
            np.unravel_index(agent_pos, self.world_csv.shape)[::-1] + np.array( [0.5, 0.5]), radius=0.1,
            facecolor="red"))

    def show_action_value(self):
        leg = []
        v_max = -1
        v_min = 1
        color_list = ["r", "b", "g", "orange", "m"]
        action_list =["^", "<", ">", "v", "o"]
        valid_action = [a for a in range(self.collaborate_state)
                        if a in self.collaborate.a_vector_a[self.collaborate_state]]
        for a in valid_action:
            v = np.array([self.collaborate.value_a(self.collaborate_state, a, self.b[i]) for i in range(len(self.b))])
            v_max = np.max([v_max, np.max(v)])
            v_min = np.min([v_min, np.min(v)])
            leg.append(plt.plot(self.b[:, 0], v, color=color_list[a])[0])
        plt.legend(leg, [action_list[a] for a in valid_action])
        plt.plot([self.belief[0], self.belief[0]], [v_max, v_min])
        plt.show()

    def show_belief(self):
        plt.ylim(0, 1)
        plt.bar(np.arange(len(self.belief)), self.belief, color="orange")

    def update_belief(self, a, s, n_s):
        self.belief *= self.collaborate.tx[:, a, s, n_s]
        # print self.belief
        # print self.collaborate.tx[:, a, s, n_s]
        self.belief /= np.sum(self.belief)

    def move(self, event):
        if event.key in self.key_a_map:
            prev_state = self.collaborate_state
            s_a = self.collaborate.get_best_action(self.collaborate_state, self.belief)
            self.collaborate_state = self.collaborate.baseMDP.get_next_state(s_a, self.key_a_map[event.key], prev_state)
            self.update_belief(s_a, prev_state, self.collaborate_state)

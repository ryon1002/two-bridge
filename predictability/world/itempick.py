import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import squareform, pdist
import actiontype
import rewardtype
from predictability.lib import probutil


class ItemPickWorld(object):
    def __init__(self, size, items, human, agent, reward=True):
        self.map = np.zeros(size)

        self.items = [i[0] for i in items]
        self.human = np.array([human])
        self.agent = np.array([agent])
        self.dists = squareform(pdist(np.array(self.items + list(self.human) + list(self.agent)),
                                      metric="cityblock"))

        self.item_property = [i[1] for i in items]

        self.assign_action = actiontype.AssignAction(self)
        self.reward = rewardtype.RewardWeight(self, reward)

        self._r_assign_reward = None

    @property
    def r_assign_reward(self):
        if self._r_assign_reward is None:
            self.r_assign_reward = "Dummy"
        return self._r_assign_reward

    @r_assign_reward.setter
    def r_assign_reward(self, _):
        self._r_assign_reward = np.array(
            [[self._ordered_assign_cost(a, r) for a in self.assign_action.all_action_seq]
             for r in self.reward.get_all_rewards()]).T

    def _prob_assign__reward_dist(self, reward):  # P(A | r)
        return np.array([self.get_reward(a, reward) for a in self.assign_action.all_action_seq])

    def _ordered_assign_cost(self, assigns, weight):
        return min(self._single_ordered_assign_cost(assigns.assign[0], weight[0]),
                   self._single_ordered_assign_cost(assigns.assign[1], weight[1]))

    def _single_ordered_assign_cost(self, assign, weight):
        return -sum([self.dists[assign[i], assign[i + 1]] for i in range(len(assign) - 1)]) \
               - np.sum(weight[list(assign)])

    def show_world(self, path=()):
        color_map = {0: "lightblue", 1: "red"}

        for y, x in itertools.product(*[range(i) for i in self.map.shape]):
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))
        for i, (y, x) in enumerate(self.items):
            plt.gca().add_patch(
                patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor=color_map[self.item_property[i]],
                               edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.human[0, 1] + 0.5, self.human[0, 0] + 0.5),
                                           0.4, facecolor="lightgreen", edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.agent[0, 1] + 0.5, self.agent[0, 0] + 0.5),
                                           0.4, facecolor="pink", edgecolor="k"))
        plt.ylim(0, self.map.shape[0])
        plt.xlim(0, self.map.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()


class ItemPickWorldItem(ItemPickWorld):
    def __init__(self, size, items, human, agent, reward=True):
        super(ItemPickWorldItem, self).__init__(size, items, human, agent, reward)
        self.action_type = actiontype.AssignAction(self)

    def prob_reward__action_dist(self, beta=1):  # P(r | a)
        p_r__assign = probutil.make_prob_2d_dist(self.r_assign_reward, 0, beta)  # P(r | As)
        prob = np.array([np.sum(p_r__assign[self.action_type.get_action_seq_index(c)], axis=0)
                         for c in self.assign_action.get_all_condition(1)])
        return probutil.normalized_2d_array(prob, 1)

    def prob_reward__action_dist2(self, beta=1):  # P(r | a)
        p_r__assign = probutil.make_prob_2d_dist(self.r_assign_reward, 0, beta)  # P(r | As)
        p_assign__r = probutil.normalized_2d_array(p_r__assign, 1)  # p(As | r)
        prob = np.zeros((p_assign__r.shape[0], len(self.items)))
        for i, c in enumerate(self.assign_action.get_all_condition(1)):
            index = self.action_type.get_action_seq_index(c)
            p = probutil.normalized_2d_array(p_r__assign[index], axis=0)
            p = np.sum(p, axis=1) / 3
            prob[index, i] = p
        return np.dot(prob.T, p_assign__r)


class ItemPickWorldMove(ItemPickWorld):
    def __init__(self, size, items, human, agent, reward=True):
        super(ItemPickWorldMove, self).__init__(size, items, human, agent, reward)
        self.action_id = {0: np.array([1, 0]), 1: np.array([0, -1]), 2: np.array([0, 1]),
                          3: np.array([-1, 0])}
        self.action_type = actiontype.AssignAction(self)

    def _cost_after_action(self, pos, action, item):
        return np.sum(np.abs(pos + self.action_id[action] - np.array(self.items[item])))

    def prob_subgoal__action(self, world, beta=1):
        p_action__subgoal = np.empty((len(world.items), len(world.action_id)))
        for i in range(len(world.items)):
            costs = -np.array(
                [self._cost_after_action(world.agent, a, i) for a in range(len(world.action_id))])
            p_action__subgoal[i] = probutil.make_prob_dist(costs, beta)
        return probutil.normalized_2d_array(p_action__subgoal, 0)

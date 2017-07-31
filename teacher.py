import numpy as np
import itertools
from agent import Agent


class Teacher(object):
    def __init__(self, world, max_length=12):
        self.prob_limit = 0.00000001
        self.reward_candidate = [p for p in itertools.permutations((-10, -0.1, -0.01))]
        # self.reward = self.reward_candidate[0]
        self.reward_id = 1
        # print self.reward_candidate[self.reward_id]

        beta = 0.5
        self.calc_policies(beta, max_length, world)

        # path = ([5, 13, 14, 22, 30, 31, 39, 38, 46, 45, 53, 61], [0, 2, 0, 0, 2, 0, 1, 0, 1, 0, 0])
        #
        # print "test"
        # world.set_reward(self.reward_candidate[0])
        # agent = Agent.Agent(world)
        # agent.calc_policy("softmax", beta=beta)
        #
        # print agent.q[:, path[0]]
        # print agent.policy[:, path[0]]
        #
        # world.set_reward(self.reward_candidate[1])
        # agent = Agent.Agent(world)
        # agent.calc_policy("softmax", beta=beta)
        # print agent.q[:, path[0]]
        # print agent.policy[:, path[0]]
        #
        # print [self.limit_policies[0, path[1][i], path[0][i]] for i in range(len(path[1]))]
        # print [self.limit_policies[1, path[1][i], path[0][i]] for i in range(len(path[1]))]
        # exit()

    def calc_policies(self, beta, max_length, world):
        self.policies = np.zeros((len(self.reward_candidate), max_length, world.a, world.s))
        self.limit_policies = np.zeros((len(self.reward_candidate), world.a, world.s))
        for r_id, reward in enumerate(self.reward_candidate):
            world.set_reward(reward)
            self.agent = Agent.Agent(world)
            for l in range(max_length):
                self.agent.calc_policy("max_q_softmax", l, beta=beta)
                self.policies[r_id, l] = self.agent.policy
            self.agent.calc_policy("max_q_softmax", beta=beta)
            self.limit_policies[r_id] = self.agent.policy

    def get_teach_path(self, start_state, type):
        path_candidate = [g for g in self.get_path_candidate(start_state, 12, 1.0)]
        path_prob = np.array([self.calc_prob(p) for p in path_candidate])
        if type == "all_probability":
            path_id = np.argmax(path_prob[:, self.reward_id])
            return path_candidate[path_id][0]
        if type == "individual_count":
            path_prob = (path_prob / np.sum(path_prob, axis=0))
            print path_prob[:, self.reward_id]
            exit()
        else:
            raise ValueError("No implemented method.")
        # print path_candidate
        # # path_probs = np.array([self.calc_prob(p) for p in path_candidate])
        # # print path_probs
        # print path_id
        # # for i in range(len(path_candidate)):
        # #     print i, path_candidate[i]
        # exit()
        # return path_candidate[path_id][0]
        # exit()
        # return self.agent.calc_policy_and_move("greedy", start_state)[0]

    def get_path_candidate(self, start_state, length, prob):
        if length == 0 or self.agent.world.is_terminate(start_state):
            yield [start_state], []
            return

        for a in range(4):
            tmp_prob = prob * self.policies[self.reward_id, length - 1, a, start_state]
            if tmp_prob > self.prob_limit:
                state = self.agent.world.get_next_state(a, start_state)
                for s_succesor, a_succesor in self.get_path_candidate(state, length - 1, tmp_prob):
                    if self.agent.world.is_terminate(s_succesor[-1]):
                        # if start_state not in s_succesor:
                        yield [start_state] + s_succesor, [a] + a_succesor

    def calc_prob(self, path):
        # prob = [np.prod([policy[i, path[1][i], path[0][i]] for i in range(len(path[1]))])
        #         for policy in self.policies]
        prob = [np.prod([policy[path[1][i], path[0][i]] for i in range(len(path[1]))])
                for policy in self.limit_policies]
        prob /= np.sum(prob)
        return prob

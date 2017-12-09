import math
import numpy as np
import MDP


class CombinedMDP(MDP.MDP):
    def __init__(self, mdp1, mdp2, mdp1_first=True, **kwargs):
        self.mdp_s = [mdp1.s, mdp2.s]
        self.mdp_a = [mdp1.a, mdp2.a]
        self.mdp1_first = mdp1_first
        super(CombinedMDP, self).__init__(self.mdp_s[0] * self.mdp_s[1], self.mdp_a[0] * self.mdp_a[1], **kwargs)
        if self.sparse:
            for m1_s in range(self.mdp_s[0]):
                for m1_a in range(self.mdp_a[0]):
                    for m1_ns in np.where(mdp1.t[m1_a, m1_s] > 0)[0]:
                        for m2_s in range(self.mdp_s[1]):
                            for m2_a in range(self.mdp_a[1]):
                                ns = self.get_state(m1_ns, np.arange(self.mdp_s[1]))
                                self.t[self.get_action(m1_a, m2_a)][self.get_state(m1_s, m2_s), ns] \
                                    = mdp1.t[m1_a, m1_s, m1_ns] * mdp2.t[m2_a, m2_s]
        else:
            for m1_s in range(self.mdp_s[0]):
                for m1_a in range(self.mdp_a[0]):
                    for m1_ns in np.where(mdp1.t[m1_a, m1_s] > 0)[0]:
                        for m2_s in range(self.mdp_s[1]):
                            for m2_a in range(self.mdp_a[1]):
                                ns = self.get_state(m1_ns, np.arange(self.mdp_s[1]))
                                self.t[self.get_action(m1_a, m2_a), self.get_state(m1_s, m2_s), ns] \
                                    = mdp1.t[m1_a, m1_s, m1_ns] * mdp2.t[m2_a, m2_s]

    def get_state(self, mdp1_state, mdp2_state, reverse=False):
        if reverse:
            mdp1_state, mdp2_state = mdp2_state, mdp1_state
        if self.mdp1_first:
            return mdp1_state + mdp2_state * self.mdp_s[0]
        return mdp2_state + mdp1_state * self.mdp_s[1]

    def get_action(self, mdp1_action, mdp2_action, reverse=False):
        if reverse:
            mdp1_action, mdp2_action = mdp2_action, mdp1_action
        if self.mdp1_first:
            return mdp1_action + mdp2_action * self.mdp_a[0]
        return mdp2_action + mdp1_action * self.mdp_a[1]

    def get_individual_state(self, state, reverse=False):
        if reverse:
            return int(math.floor(state / self.mdp_s[1])), state % self.mdp_s[1]
        return state % self.mdp_s[0], int(math.floor(state / self.mdp_s[0]))

    def get_next_state(self, mdp1_action, mdp2_action, state):
        return super(CombinedMDP, self).get_next_state(self.get_action(mdp1_action, mdp2_action), state)

    def get_transition_with_others_policy(self, index, policy):
        other = 1 - index
        t = np.zeros((self.mdp_a[index], self.s, self.s))
        for m2_s in range(self.mdp_s[other]):
            for m1_a in range(self.mdp_a[index]):
                a = self.get_action(m1_a, np.arange(self.mdp_a[other]), index != 0)
                s = self.get_state(np.arange(self.mdp_s[index]), m2_s, index != 0)
                t[m1_a, s, :] = np.sum(self.t[a][:, s] * policy[:, [m2_s]][:, :, np.newaxis], axis=0)
        return t

    def get_tr_with_others_policy(self, index, policy):
        other = 1 - index
        t = np.zeros((self.mdp_a[index], self.s, self.s))
        r = np.zeros((self.mdp_a[index], self.s, self.s))
        for s in range(self.s):
            for m_a in range(self.mdp_a[index]):
                a = self.get_action(m_a, np.arange(self.mdp_a[other]), index != 0)
                print m_a, a, s
                print t.shape
                print policy.shape
                t[m_a, s, :] = np.sum(self.t[a, s] * policy[:,s][:, np.newaxis], axis=0)
                r[m_a, s, :] = np.sum(self.r[a, s] * policy[:,s][:, np.newaxis], axis=0)
        return t, r

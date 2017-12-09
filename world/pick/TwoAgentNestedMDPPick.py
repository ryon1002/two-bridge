import numpy as np
from world.core import TwoAgentNestedMDP


class TwoAgentNestedMDPPick(TwoAgentNestedMDP.TwoAgentNestedMDP):
    def __init__(self, mdp1, mdp2, **kwargs):
        super(TwoAgentNestedMDPPick, self).__init__(mdp1, mdp2, **kwargs)
        # self.solver = ValueIteration.ValueIteration()
        # super(TwoAgentNestedMDP, self).__init__(self.baseMDP.s, len(other_policies), self.baseMDP.mdp_a[main_index])
        # for y, policy in enumerate(other_policies):
        #     self.tx[y] = self.baseMDP.get_transition_with_others_policy(main_index, policy)

    # def add_sink_state(self, main_state, target_state):
    #     self.s = self.s + 1
    #     tmp_t = self.t
    #     self.t = np.zeros((self.a, self.s, self.s))
    #     self.t[:, :-1, :-1] = tmp_t
    #     for a in range(self.a):
    #         self.t[a, self.get_state(main_state, target_state), 0:-1] = 0
    #         self.t[a, self.get_state(main_state, target_state), -1] = 1
    #     self.t[:, -1, -1] = 1

    # def calc_policy(self, main_index, other_policy=None):
    #     if other_policy is None:
    #         other_policy = np.ones((self.mdp_a[1 - main_index], self.s))
    #         other_policy /= np.sum(other_policy, axis=0)
    #     t, r = self.get_tr_with_others_policy(main_index, other_policy)
    #
    #     mdp = MDP.MDP(t.shape[1], t.shape[0], self.d)
    #     mdp.t = t
    #     mdp.r = r
    #     return self.solver.get_greedy_policy(mdp)

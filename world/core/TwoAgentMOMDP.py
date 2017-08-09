import numpy as np
import MOMDP
import CombinedMDP


class TwoAgentMOMDP(MOMDP.MOMDP):
    def __init__(self, mdp1, mdp2, main_index, other_policies):
        self.baseMDP = CombinedMDP.CombinedMDP(mdp1, mdp2)
        super(TwoAgentMOMDP, self).__init__(self.baseMDP.s, len(other_policies), self.baseMDP.mdp_a[main_index])
        for y, policy in enumerate(other_policies):
            self.tx[y] = self.baseMDP.get_transition_with_others_policy(main_index, policy)

    def add_sink_state(self, main_state, target_state):
        self.x = self.x + 1
        tmp_tx = self.tx
        self.tx = np.zeros((self.y, self.a, self.x, self.x))
        self.tx[:, :, :-1, :-1] = tmp_tx
        for y in range(self.y):
            for a in range(self.a):
                self.tx[y, a, self.baseMDP.get_state(main_state, target_state), 0:-1] = 0
                self.tx[y, a, self.baseMDP.get_state(main_state, target_state), -1] = 1
            self.tx[y, :, -1, -1] = 1

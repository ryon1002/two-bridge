import numpy as np
from core import TwoAgentMOMDP


class TwoAgentMOMDPBridge(TwoAgentMOMDP.TwoAgentMOMDP):
    def __init__(self, agents, sidekick, bridge, goal):  # depth=1, bs=None):
        self.mainFirst = True
        super(TwoAgentMOMDPBridge, self).__init__(sidekick, agents[0], 0, [a.aProb for a in agents])
        self.add_sink_state(np.arange(sidekick.s), agents[0].shrinkMap[goal])

        self.ty = np.zeros((self.y, self.a, self.x, self.y))
        for y in range(self.y):
            self.ty[y, :, :, y] = 1

        self.r = np.zeros((self.a, self.x, self.y))
        goal = agents[0].shrinkMap[goal]
        self.r[:4, :, :] = -0.01
        self.r[:, -1, :] = 0
        for a in range(self.a):
            self.r[a, self.baseMDP.get_state(np.arange(sidekick.s), goal)] = 1
            for b in bridge:
                a_b = agents[0].shrinkMap[b]
                self.r[a, self.baseMDP.get_state(np.arange(sidekick.s), a_b), :] = -0.4
                self.r[a, self.baseMDP.get_state(sidekick.shrinkMap[b], a_b), :] = -0.01

        self.pre_culc()

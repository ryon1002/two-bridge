import numpy as np
from core import TwoAgentNestedMDP
import FieldMDP


class TwoAgentNestedMDPBridge(TwoAgentNestedMDP.TwoAgentNestedMDP):
    def __init__(self, world_csv, agent_state, sidekick_state, bridge, goal):
        self.bridge = bridge

        self.agent = FieldMDP.FieldMDP(world_csv)
        for b in self.bridge:
            self.agent.make_road(b)
        self.agent.shrink_state(agent_state)
        # self.agent.set_goal(self.agent.shrinkMap[goal])

        self.sidekick = FieldMDP.FieldMDP(1 - world_csv)
        for b in self.bridge:
            self.sidekick.set_goal(b)
        self.sidekick.shrink_state(sidekick_state)

        super(TwoAgentNestedMDPBridge, self).__init__(self.sidekick, self.agent)
        self.add_sink_state(np.arange(self.sidekick.s), self.agent.shrinkMap[goal])

        self.r = np.zeros_like(self.t)
        self.r[:, :, self.get_state(np.arange(self.sidekick.s), self.agent.shrinkMap[goal])] = 1
        for b in self.bridge:
            self.r[:, :, self.get_state(np.arange(self.sidekick.s), self.agent.shrinkMap[b])] = -0.4
            self.r[:, :, self.get_state(self.sidekick.shrinkMap[b], self.agent.shrinkMap[b])] = 0
        self.r[:, :, -1] = 0

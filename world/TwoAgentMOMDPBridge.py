import numpy as np
from core import TwoAgentMOMDP
from world import FieldMDP


class TwoAgentMOMDPBridge(TwoAgentMOMDP.TwoAgentMOMDP):
    def __init__(self, world_csv, agent_state, sidekick_state, bridge, goal, beta):  # depth=1, bs=None):
        self.bridge = bridge
        self.agents = []
        for i in range(len(self.bridge)):
            agent = FieldMDP.FieldMDP(world_csv)
            for j in range(len(self.bridge)):
                if i == j:
                    agent.make_road(self.bridge[j])
                else:
                    agent.make_road(self.bridge[j], -0.2)
            agent.shrink_state(agent_state)
            self.agents.append(agent)
            agent.set_goal(agent.shrinkMap[goal])
            agent.do_value_iteration()
            agent.aProb = agent.softmax(agent.q, beta)

        self.sidekick = FieldMDP.FieldMDP(1 - world_csv)
        for b in self.bridge:
            self.sidekick.set_goal(b)
        self.sidekick.shrink_state(sidekick_state)

        super(TwoAgentMOMDPBridge, self).__init__(self.sidekick, self.agents[0], 0, [a.aProb for a in self.agents])
        self.add_sink_state(np.arange(self.sidekick.s), self.agents[0].shrinkMap[goal])

        self.ty = np.zeros((self.y, self.a, self.x, self.y))
        for y in range(self.y):
            self.ty[y, :, :, y] = 1

        self.r = np.zeros((self.a, self.x, self.y))
        goal = self.agents[0].shrinkMap[goal]
        self.r[:4, :, :] = -0.01
        self.r[:, -1, :] = 0
        for a in range(self.a):
            self.r[a, self.baseMDP.get_state(np.arange(self.sidekick.s), goal)] = 1
            for b in bridge:
                a_b = self.agents[0].shrinkMap[b]
                self.r[a, self.baseMDP.get_state(np.arange(self.sidekick.s), a_b), :] = -0.4
                self.r[a, self.baseMDP.get_state(self.sidekick.shrinkMap[b], a_b), :] = -0.01

        self.pre_culc()

    def get_state(self, sidekick_pos, agent_pos):
        return self.baseMDP.get_state(self.sidekick.shrinkMap[sidekick_pos], self.agents[0].shrinkMap[agent_pos])

    def get_pos(self, state):
        sidekick_state, agent_state = self.baseMDP.get_individual_state(state)
        return self.sidekick.i_shrinkMap[sidekick_state], self.agents[0].i_shrinkMap[agent_state]

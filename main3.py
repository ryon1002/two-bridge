import numpy as np
from world import TwoAgentNestedMDPBridge

np.set_printoptions(edgeitems=3200, linewidth=10000, precision=6)

bridge = [15, 19]
agent_pos = 2
sidekick_pos = 22
goal = 32
beta = 50

world_csv = np.loadtxt("world/map/map.csv", delimiter=",")
mdp = TwoAgentNestedMDPBridge.TwoAgentNestedMDPBridge(world_csv, agent_pos, sidekick_pos, bridge, goal)
policy = mdp.calc_policy(1)
# print policy.T
print mdp.calc_policy(0, policy).T

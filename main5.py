import Application2
import numpy as np
from world import HierarchicalItemPickWorld
from world.pick import OneItemPickWorld
from world.pick import TwoAgentNestedMDPPick

np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)
world_shape = (5, 5)
world1 = OneItemPickWorld.OneItemPickWorld(world_shape, (3, 2))
world2 = OneItemPickWorld.OneItemPickWorld(world_shape, (3, 2))
# world = TwoAgentNestedMDP.TwoAgentNestedMDP(world1, world2, sparse=True)
world = TwoAgentNestedMDPPick.TwoAgentNestedMDPPick(world1, world2)
# world.add_sink_state(17, 17)

from solver import ValueIteration
solver = ValueIteration.ValueIteration()
policy = solver.get_greedy_policy(world1)
print world.calc_policy(0, policy)

# print world.t
# world.show_world()
# world = HierarchicalItemPickWorld.HierarchicalItemPickWorld()
# policy = world.calc_policy()

# agent = Agent.Agent()
# path = agent.move(world, policy, 17)

# print [p % world.base_s for p in path[0]]

# world.show_world()
# app = Application2.Application()

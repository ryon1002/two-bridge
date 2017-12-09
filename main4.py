import numpy as np
from world import MultipleItemPickWorld
from agent import Agent

np.set_printoptions(edgeitems=3200, linewidth=10000, precision=3)

world = MultipleItemPickWorld.MultipleItemPickWorld()
policy = world.calc_policy()

agent = Agent.Agent()
path = agent.move(world, policy, 17)

print [p % world.base_s for p in path[0]]

world.show_world(path[0])

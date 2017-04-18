# from world import TwoAgentMDP
import Application2
# import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)

# worldmap = np.loadtxt("world/map2.csv", delimiter=",")
# world = TwoAgentMDP.TwoAgentMDP(worldmap)
# world.setAgent(2)
# world.setSidekick(59)
# estimator = Estimator.Estimator(worldmap)
# estimator2 = Estimator2.Estimator(worldmap)
# sidekick = Sidekick.Sidekick(world, worldmap, estimator, estimator2)
app = Application2.Application("world/map2.csv")

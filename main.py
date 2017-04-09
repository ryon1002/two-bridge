from world import FieldMDP
import Estimator, Estimator2, Sidekick, Application
# import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(edgeitems=3200, linewidth=1000, precision=6)

worldmap = np.loadtxt("world/map.csv", delimiter=",")
world = FieldMDP.FieldMDP(worldmap)
world.setAgent(10)
world.setSidekick(59)
estimator = Estimator.Estimator(worldmap)
estimator2 = Estimator2.Estimator(worldmap)
sidekick = Sidekick.Sidekick(world, worldmap, estimator, estimator2)
app = Application.Application(world, estimator, estimator2, sidekick)

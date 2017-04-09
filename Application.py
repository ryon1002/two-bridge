import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class Application():
    def __init__(self, world, estimator, estimator2, sidekick):
        self.fig = plt.figure() 
        self.world = world
        self.estimator = estimator
        self.estimator2 = estimator2
        self.sidekick = sidekick
        self.fig.canvas.mpl_connect('key_press_event', self.keyDown)
        self.draw()
        plt.show()
    
    def keyDown(self, event):
        sa = self.world.move(event)
        if sa is not None:
            self.estimator.updateBelief(sa[0], sa[1])
#             s, a = self.sidekick.doBestAction()
            s, a = self.sidekick.doBestAction2()
            self.estimator2.updateBelief(s, a)
            self.draw()

    def draw(self):
        plt.clf()
        G = gridspec.GridSpec(3, 2)
        self.fig.add_subplot(G[:, 0])
        self.world.showWorld()
        self.fig.add_subplot(G[0, 1])
        self.estimator.showBelief()
        self.fig.add_subplot(G[1, 1])
        self.estimator2.showBelief()
        self.fig.add_subplot(G[2, 1])
        self.sidekick.showBelief() 
        self.fig.canvas.draw()

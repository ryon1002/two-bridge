import itertools
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import predictability.world.itempick


class ItemPickWorldProperty(predictability.world.itempick.ItemPickWorld):

    def show_world(self, path=()):
        for y, x in itertools.product(*[range(i) for i in self.map.shape]):
            plt.gca().add_patch(patches.Rectangle((x, y), 1, 1, facecolor="w", edgecolor="k"))
        for i, (y, x) in enumerate(self.items):
            if self.item_property[i] == 0:
                plt.gca().add_patch(
                    patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor="lightblue", edgecolor="k"))
            else:
                plt.gca().add_patch(
                    patches.Circle((x + 0.5, y + 0.5), 0.4, facecolor="red", edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.human[1] + 0.5, self.human[0] + 0.5),
                                           0.4, facecolor="lightgreen", edgecolor="k"))
        plt.gca().add_patch(patches.Circle((self.agent[1] + 0.5, self.agent[0] + 0.5),
                                           0.4, facecolor="pink", edgecolor="k"))
        plt.ylim(0, self.map.shape[0])
        plt.xlim(0, self.map.shape[1])
        plt.tick_params(labelbottom="off", labelleft="off")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

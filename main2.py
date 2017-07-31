from world import ItemPickWorld
import teacher
import numpy as np


np.set_printoptions(edgeitems=3200, linewidth=1000, precision=3)

world_csv = np.loadtxt("world/map/pick_world.csv", delimiter=",", dtype=np.int)
item_csv = np.loadtxt("world/map/pick_item.csv", delimiter=",")
item_world = ItemPickWorld.ItemPickWorld(world_csv, item_csv)
# teach = teacher.Teacher(item_world, 12)
# path =teach.get_teach_path(37)

teach = teacher.Teacher(item_world, 20)
# path = teach.get_teach_path(5, "all_probability")
path = teach.get_teach_path(5, "individual_count")
print path
item_world.show_world(path)

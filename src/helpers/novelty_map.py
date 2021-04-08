import numpy as np

class NoveltyMap():
    """
    Implements a map as used in MAP-Elites.
    Each cell in the n-dimensional grid represents solutions whose behaviour characterisation falls within a certain
    range.
    Each cell contains just one elite i.e. a solution that has higher fitness than any of the others that have occupied
    that grid cell
    """
    def __init__(self, shape_tuple, seed):
        #TODO: Don't hardcode shape
        #TODO: Don't hardcode BC in get and set
        self.shape = shape_tuple
        self.map = np.array([[None for x in range(shape_tuple[0])] for y in range(shape_tuple[1])])
        self.np_random = np.random.default_rng(seed)

    def get_elite(self, behaviour_characterisation):
        x = int(behaviour_characterisation[0])
        y = int(behaviour_characterisation[1])
        return self.map[x][y]

    def get_random_elite(self):
        elites_list = self.map.flatten()
        elites_list = elites_list[elites_list != None]
        return self.np_random.choice(elites_list)

    def set_elite(self, elite):
        behaviour_characterisation = elite.behaviour_characterisation
        x = int(behaviour_characterisation[0])
        y = int(behaviour_characterisation[1])
        self.map[x][y] = elite


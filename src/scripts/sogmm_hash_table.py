import numpy as np


class GMMSpatialHash:
    def __init__(self, width=100, height=100, depth=100, resolution=0.2):
        self.w = (int)(width)
        self.h = (int)(height)
        self.d = (int)(depth)
        self.res = resolution
        self.o = (-1.0 * resolution / 2.0) * np.array([width, height, depth])

        self.hash_table = {}

    def add_point(self, key, value):
        idx = self.point_to_index(key)

        try:
            temp = self.hash_table[idx]
        except KeyError:
            self.hash_table[idx] = list()

        self.hash_table[idx].append(value)

    def add_points(self, points, values):
        N = points.shape[0]
        for i in range(N):
            self.add_point(points[i, :3], values[i])

    def point_to_index(self, point):
        r = (int)((point[1] - self.o[1]) / self.res)
        c = (int)((point[0] - self.o[0]) / self.res)
        s = (int)((point[2] - self.o[2]) / self.res)

        return (r * self.w + c) * self.d + s

    def find_point(self, key):
        idx = self.point_to_index(key)

        try:
            temp = self.hash_table[idx]
            return temp
        except KeyError:
            return [-1]

    def find_points(self, points):
        N = points.shape[0]
        results = []
        for i in range(N):
            results += self.find_point(points[i, :3])

        results = np.array(results, dtype=int)

        return np.unique(results[results >= 0])

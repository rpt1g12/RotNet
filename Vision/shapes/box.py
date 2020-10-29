import numpy as np
import sys
from matplotlib.patches import Rectangle, Patch
from functools import reduce

from Vision.shapes import Shape


class Box(Shape):
    """
    Clase para regiones con forma de caja o rectangulo
    """

    def __init__(self,
                 x_min: float, y_min: float, width: float, height: float, theta: float = 0):
        super(Box, self).__init__()
        self.x_min = x_min
        self.y_min = y_min
        self.width = width
        self.height = height
        self.theta = np.deg2rad(theta)  # In radians
        # Lenght of the box diagonal
        self.diagonal = np.sqrt(height ** 2 + width ** 2)
        # Angle between diagonals
        self.alpha = np.arctan(height / width)

    @classmethod
    def from_minmax(cls, x_min, y_min, x_max, y_max, theta=0.0):
        theta_rad = np.deg2rad(theta)
        sin = np.sin(theta_rad)
        cos = np.cos(theta_rad)
        cos2 = np.cos(2 * theta_rad)
        width_p = (x_max - x_min)
        height_p = (y_max - y_min)
        width = (width_p * cos - height_p * sin) / cos2
        height = (height_p * cos - width_p * sin) / cos2
        return cls(x_min, y_min, width, height, theta)

    @classmethod
    def from_centdim(cls, x0, y0, width, height, theta=0.0):
        theta_rad = np.deg2rad(theta)
        sin = np.sin(theta_rad)
        cos = np.cos(theta_rad)
        x_min = x0 - 0.5 * (width * cos - height * sin)
        y_min = y0 - 0.5 * (width * sin + height * cos)
        return cls(x_min, y_min, width, height, theta)

    def get_centroid(self) -> tuple:
        theta = self.theta
        sin = np.sin(theta)
        cos = np.cos(theta)
        x0 = self.x_min + 0.5 * (self.width * cos - self.height * sin)
        y0 = self.y_min + 0.5 * (self.width * sin + self.height * cos)
        return x0, y0

    def get_sw(self):
        return self.x_min, self.y_min

    def get_se(self):
        alpha = self.alpha
        theta = self.theta
        x0, y0 = self.get_centroid()
        half_diag = self.diagonal * 0.5
        x = x0 + half_diag * np.cos(alpha - theta)
        y = y0 - half_diag * np.sin(alpha - theta)
        return x, y

    def get_nw(self):
        alpha = self.alpha
        theta = self.theta
        x0, y0 = self.get_centroid()
        half_diag = self.diagonal * 0.5
        x = x0 - half_diag * np.cos(alpha - theta)
        y = y0 + half_diag * np.sin(alpha - theta)
        return x, y

    def get_ne(self):
        alpha = self.alpha
        theta = self.theta
        x0, y0 = self.get_centroid()
        half_diag = self.diagonal * 0.5
        x = x0 + half_diag * np.cos(alpha + theta)
        y = y0 + half_diag * np.sin(alpha + theta)
        return x, y

    def get_vertices(self):
        return [self.get_sw(), self.get_se(), self.get_nw(), self.get_ne()]

    def get_patch(self) -> Patch:
        patch = Rectangle((self.x_min, self.y_min), self.width, self.height, np.rad2deg(self.theta))
        return patch

    def get_left(self) -> int:
        vs = self.get_vertices()
        left = reduce(lambda x0, x1: min(x0, x1[0]), vs, sys.float_info.max)
        return int(left)

    def get_right(self) -> int:
        vs = self.get_vertices()
        right = reduce(lambda x0, x1: max(x0, x1[0]), vs, sys.float_info.min)
        return int(np.ceil(right))

    def get_bottom(self) -> int:
        vs = self.get_vertices()
        bottom = reduce(lambda x0, x1: min(x0, x1[1]), vs, sys.float_info.max)
        return int(bottom)

    def get_top(self) -> int:
        vs = self.get_vertices()
        top = reduce(lambda x0, x1: max(x0, x1[1]), vs, sys.float_info.min)
        return int(np.ceil(top))

    def get_bounds(self) -> tuple:
        return self.get_left(), self.get_bottom(), self.get_right(), self.get_top()

    def get_height(self) -> float:
        return self.height

    def get_width(self) -> float:
        return self.width

    def shift_origin(self, dx: float, dy: float) -> None:
        self.x_min += dx
        self.y_min += dy

    def affine(self, M: np.ndarray) -> None:
        sw = (*self.get_sw(), 1)
        se = (*self.get_se(), 1)
        nw = (*self.get_nw(), 1)
        new_coords = (np.dot(M, np.array([sw, se, nw]).transpose())).transpose()
        self.compute_from_coords(new_coords)
        pass

    def resize(self, scale: tuple) -> None:
        vertices = self.get_vertices()
        new_vertices = [[p*s for p, s in zip(vertex, scale)] for vertex in vertices]
        self.compute_from_coords(np.array(new_vertices))
        pass

    def compute_from_coords(self, coordinates: np.array):
        self.x_min = coordinates[0, 0]
        self.y_min = coordinates[0, 1]
        bottom_edge = coordinates[1] - coordinates[0]
        left_edge = coordinates[2] - coordinates[0]
        self.theta = np.arctan(bottom_edge[1] / bottom_edge[0])
        width = np.linalg.norm(bottom_edge)
        height = np.linalg.norm(left_edge)
        self.width = width
        self.height = height
        # Lenght of the box diagonal
        self.diagonal = np.sqrt(height ** 2 + width ** 2)
        # Angle between diagonals
        self.alpha = np.arctan(height / width)
from typing import List, Tuple

import numpy as np
import sys
from matplotlib.patches import Patch
from matplotlib.patches import Polygon as _Polygon
from functools import reduce

from Vision.shapes import Shape


class Polygon(Shape):
    """
    Clase para regiones con forma de poligono de n vertices
    """

    def __init__(self, vertices: List[Tuple[float, float]]):
        """
        Constructor a partir de lista de tuplas con cordenadas x e y
        :param vertices: Lista de tuplas con coordenadas x e y
        """
        super(Polygon, self).__init__()
        self.vertices = vertices

    def get_centroid(self) -> tuple:
        x_0 = 0
        y_0 = 0
        n_vertices = len(self.vertices)
        for vertex in self.vertices:
            x_0 += vertex[0]
            y_0 += vertex[1]
        return x_0 / n_vertices, y_0 / n_vertices

    def get_patch(self) -> Patch:
        return _Polygon(self.vertices, True)

    def get_mask_array(self, loc_array: np.array=None) -> np.array:
        """
        Obtener matriz de la mascara para segmentation
        :param loc_array: matriz con las posiciones de los pixeles de la imagen para poner la mascara
        :return: mascara de 0/1 en una matriz del mismo tamaño que la imagen
        """
        patch = self.get_patch()
        if loc_array is None:
            max_patch = np.array(self.vertices).max(initial=20) + 1
            loc_array = np.array([[(i, j) for i in range(max_patch)] for j in range(max_patch)])
        mask_shape = loc_array.shape
        mask = patch.contains_points(loc_array.reshape(-1, 2),
                                     radius=-0.5).reshape(mask_shape[0], mask_shape[1])
        return mask

    def get_vertices(self):
        return self.vertices

    def set_vertices(self, vertices):
        self.vertices = vertices

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
        return self.get_top() - self.get_bottom()

    def get_width(self) -> float:
        return self.get_right() - self.get_left()

    def shift_origin(self, dx: float, dy: float) -> None:
        new_vertices = map(lambda p: (p[0] + dx, p[1] + dy), self.vertices)
        self.vertices = list(new_vertices)

    def affine(self, M: np.ndarray) -> None:
        _vertices_expanded = list(map(lambda p: (p[0], p[1], 1), self.vertices))
        new_coords = (np.dot(M, np.array(_vertices_expanded).transpose())).transpose()
        new_vertices = map(lambda p: (p[0], p[1]), new_coords)
        self.vertices = list(new_vertices)

    def resize(self, scale: tuple) -> None:
        vertices = self.vertices
        new_vertices = [tuple(p*s for p, s in zip(vertex, scale)) for vertex in vertices]
        self.vertices = new_vertices



import abc
import copy
import numpy as np
from typing import Union

from matplotlib.patches import Patch


class Shape(metaclass=abc.ABCMeta):
    """
    Interfaz para regiones etiquetadas
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'get_centroid') and
                callable(subclass.get_centroid) and
                hasattr(subclass, 'get_left') and
                callable(subclass.get_left) and
                hasattr(subclass, 'get_right') and
                callable(subclass.get_right) and
                hasattr(subclass, 'get_top') and
                callable(subclass.get_top) and
                hasattr(subclass, 'get_bottom') and
                callable(subclass.get_bottom) and
                hasattr(subclass, 'get_height') and
                callable(subclass.get_height) and
                hasattr(subclass, 'get_width') and
                callable(subclass.get_width) and
                hasattr(subclass, 'get_bounds') and
                callable(subclass.get_bounds) and
                hasattr(subclass, 'get_patch') and
                callable(subclass.get_patch) or
                NotImplemented)

    @abc.abstractmethod
    def get_centroid(self) -> tuple:
        """
        Devuelve el centroide de la region.
        :return: Tuple que contiene las coordenadas del centroide.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_patch(self) -> Patch:
        """
        Devuelve el objeto matplotlib.patches equivalente a la region.
        :return: Un objeto matplotlib Patch
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_left(self) -> int:
        """
        Devuelve la coordenada minima en el eje horizontal de la region
        :return: Valor de la coordenada minima en el eje horizontal de
        la region en pixeles.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_right(self) -> int:
        """
        Devuelve la coordenada maxima en el eje horizontal de la region
        :return: Valor de la coordenada maxima en el eje horizontal de
        la region en pixeles.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_bottom(self) -> int:
        """
        Devuelve la coordenada minima en el eje vertical de la region
        :return: Valor de la coordenada minima en el eje vertical de
        la region en pixeles.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_top(self) -> int:
        """
        Devuelve la coordenada maxima en el eje vertical de la region
        :return: Valor de la coordenada maxima en el eje vertical de
        la region en pixeles.
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_bounds(self) -> tuple:
        """
        Devuelve la los limites de la region en una tupla
        (left, bottom, right, top)
        :return: tupla con los limites de la region en pixeles
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_height(self) -> float:
        """
        Devuelve la altura del bounding box de la region.
        :return:  Altura del bounding box de la region
        """
        raise NotImplemented

    @abc.abstractmethod
    def get_width(self) -> float:
        """
        Devuelve el ancho del bounding box de la region.
        :return:  Ancho del bounding box de la region
        """
        raise NotImplemented

    @abc.abstractmethod
    def shift_origin(self, dx: float, dy: float) -> None:
        """
        Desplaza el origen de coordenadas de la region
        :param dx: Desplazamiento horizontal
        :param dy: Desplazamiento vertical
        """
        raise NotImplemented

    @abc.abstractmethod
    def affine(self, M: np.ndarray) -> None:
        """
        Aplica una transformacion afin representada en la matriz M
        :param M: Matriz de la transformacion.
        :param scale: Escalado de la transformacion en ambas direcciones.
        Si es un unico valor, se asume escalado isotropico.
        """
        pass

    @abc.abstractmethod
    def resize(self, scale: tuple) -> None:
        """
        Aplica una transformación de escalado en las dimensiones (x,y)
        :param scale: Escalado de la transformacion en ambas direcciones.
        """
        pass

    def clone(self):
        return copy.deepcopy(self)

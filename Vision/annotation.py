import copy
import re
from matplotlib.axes import Axes
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba
from typing import List

from Vision.shapes import Shape


class Annotation(object):
    """
    Clase que contiene la informacion de una anotacion:
        - la region de la anotacion
        - la clase a la que pertenece la region
        - la confianza en la anotacion
    """

    def __init__(self, shape: Shape, cls: str, conf: float = 1.0, text=""):
        """
        Constructor de la clase.
        :param shape: Objeto Shape que delimita la region anotada.
        :param cls: Clase/Etiqueta de la region anotada.
        :param conf: Confianza en la etiqueta de la anotacion.
        """
        super(Annotation, self).__init__()
        self.shape = shape
        self.cls = cls
        self.conf = conf
        self.text = text

    def add_to_plot(self, ax: Axes = None, color: tuple = (1, 0, 0), with_confidence=True,
                    with_text=False) -> None:
        """
        Dibuja una region anotada en un eje de Matplotlib
        :param with_text: True si queremos anadir texto adicional de la anotacion.
        :param with_confidence: True si queremos anadir la confianza de la anotacion.
        :param ax: Objeto Axis de la figura donde dibujar la anotacion.
        :param color: Color de la region anotada.
        """
        if ax is None:
            ax = plt.gca()
        cls = str(self.cls)
        text = self.text
        conf = "{:1.3f}".format(self.conf)
        centroid = self.shape.get_centroid()
        nw_corner = (self.shape.get_left() + 1, self.shape.get_bottom() + 1)
        sw_corner = (self.shape.get_left() + 1, self.shape.get_top() - 1)
        patch = self.shape.get_patch()
        patch.set_fill(True)
        patch.set_edgecolor(color)
        patch.set_facecolor(to_rgba(color, 0.6))
        ax.add_patch(patch)
        ax.plot(centroid[0], centroid[1], "+", color=color)
        ax.annotate(
            cls,
            centroid,
            ha="center", va="bottom", color="k",
            weight="bold"
        )
        if with_confidence:
            ax.annotate(
                conf,
                nw_corner,
                ha="left", va="top", color="k",
                weight="bold"
            )
        if with_text:
            ax.annotate(
                text,
                sw_corner,
                ha="left", va="bottom", color="k",
                weight="bold"
            )
        pass

    def get_shape(self) -> Shape:
        return self.shape

    def get_cls(self) -> str:
        return self.cls

    def clone(self):
        return copy.deepcopy(self)


def class_filter(r_pattern: str):
    """
    Devuelve una funcion para detectar la clase de una anotacion.
    :param r_pattern: Regex para detectar la clase.
    :return: Funcion predicado que devuelve positivo cuando la clase
    de la anotacion coincide con el regex.
    """
    regex = re.compile(r_pattern)

    def predicate(annotation: Annotation) -> bool:
        return bool(regex.match(annotation.cls))

    return predicate


# Define derived types
Annotation_List = List[Annotation]

# %%

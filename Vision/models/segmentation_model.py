import abc
from typing import List

import numpy as np

from PIL import Image

from Vision.annotation import Annotation_List
from Vision import Sample


class SegmentationModel(metaclass=abc.ABCMeta):
    """
    Interfaz para Modelos de deteccion de objetos.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'predict_image_array') and
                callable(subclass.predict_image_array) or
                NotImplemented
        )

    @abc.abstractmethod
    def predict_image_array(self, img_array: np.ndarray) -> Annotation_List:
        """
        Devuelve una lista de anotaciones sobre una imagen en formato matriz de numpy
        :param img_array: Imagen en formato matriz de numpy
        :return: Lista de anotaciones predichas.
        """
        raise NotImplementedError

    def predict_image(self, img: Image) -> np.Array:
        """
        Devuelve una lista de anotaciones sobre la imagen.
        :param img: Imagen sobre la que predecir las anotaciones.
        :return: Lista de anotaciones predichas.
        """
        img_array = np.array(img)
        return self.predict_image_array(img_array)

    def predict_sample(self, sample: Sample) -> np.Array:
        """
        Devuelve una lista de anotaciones sobre la muestra.
        :param sample: Muestra sobre la que predecir las anotaciones.
        :return: Lista de anotaciones predichas.
        """
        return self.predict_image(sample.img)

    def predict_sample_append(self, sample: Sample) -> Sample:
        """
        Devuelve la muestra de entrada con anotaciones nuevas predichas.
        :param sample: Muestra de entrada
        :return: Muestra con nuevas anotaciones predichas.
        """
        annotations = self.predict_sample(sample)
        return sample.append_annotations(annotations)

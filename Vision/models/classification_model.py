import abc
from typing import Dict

import numpy as np
from PIL import Image

from Vision import Sample


class ClassificationModel(metaclass=abc.ABCMeta):
    """
    Interfaz para Modelos de clasificacion de imagenes.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'predict_image_array') and
                callable(subclass.predict_image_array) or
                NotImplemented
        )

    @abc.abstractmethod
    def predict_image_array(self, img_array: np.ndarray) -> Dict[str, float]:
        """
        Devuelve un diccionario con las etiquetas predichas y sus probalidades
        sobre una imagen en formato matriz de numpy.
        :param img_array: Imagen en formato matriz de numpy
        :return: Diccionario de etiqueta-probabilidad.
        """
        raise NotImplementedError

    def predict_image(self, img: Image) -> Dict[str, float]:
        """
        Devuelve un diccionario con las etiquetas predichas y sus probalidades sobre la imagen.
        :param img: Imagen sobre la que predecir las etiquetas.
        :return: Diccionario de etiqueta-probabilidad.
        """
        img_array = np.array(img)
        return self.predict_image_array(img_array)

    def predict_sample(self, sample: Sample) -> Dict[str, float]:
        """
        Devuelve un diccionario con las etiquetas predichas y sus probalidades
        :param sample: Muestra sobre la que predecir las etiquetas.
        :return: Diccionario de etiqueta-probabilidad.
        """
        return self.predict_image(sample.img)

    def predict_sample_append(self, sample: Sample) -> Sample:
        """
        Devuelve la muestra de entrada con etiquetas en el campo metadata
        :param sample: Muestra de entrada
        :return: Muestra con nuevas etiquetas predichas.
        """
        metadata = dict(tags=self.predict_sample(sample))
        prediction = sample.clone()
        prediction.set_metadata(metadata)
        return prediction

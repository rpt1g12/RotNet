import abc
from typing import List

import numpy as np

from PIL import Image

from Vision.annotation import Annotation_List
from Vision import Sample
from Vision.utils import list_utils


class DetectionModel(metaclass=abc.ABCMeta):
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

    def predict_image(self, img: Image) -> Annotation_List:
        """
        Devuelve una lista de anotaciones sobre la imagen.
        :param img: Imagen sobre la que predecir las anotaciones.
        :return: Lista de anotaciones predichas.
        """
        img_array = np.array(img)
        return self.predict_image_array(img_array)

    def predict_sample(self, sample: Sample) -> Annotation_List:
        """
        Devuelve una lista de anotaciones sobre la muestra.
        :param sample: Muestra sobre la que predecir las anotaciones.
        :return: Lista de anotaciones predichas.
        """
        return self.predict_image(sample.img)

    def predict_annotations(self, sample: Sample, r_pattern: str) -> List[Annotation_List]:
        """
        Devuelve una lista de listas de anotaciones que agrupa nuevas anotaciones hechas
        sobre Regiones de Interes (RoI) de la muestra. Las regiones de interes
        de la muestra son las anotaciones existente cuya clase pasa el filtro
        del regex.
        :param sample: Muestra con anotaciones.
        :param r_pattern: Filtro para las anotaciones previas.
        :return: Lista de listas con nuevas anotaciones predichas. Cada elemento de
        la lista contiene las anotaciones dentro su RoI.
        """
        img = sample.get_img()
        max_x = img.width
        max_y = img.height
        new_annotations = list()
        for roi in sample.filter_annotations(r_pattern):
            shape = roi.get_shape()
            bounds = shape.get_bounds()
            bounds = (
                max(bounds[0], 0),
                max(bounds[1], 0),
                min(bounds[2], max_x)+1,
                min(bounds[3], max_y)+1
            )
            roi_img = img.crop(bounds)
            # if the cropped area is 0 skip
            if roi_img.width*roi_img.height == 0:
                continue
            annotations = self.predict_image(roi_img)
            for ann in annotations:
                ann.get_shape().shift_origin(bounds[0], bounds[1])
            new_annotations.append(sorted(annotations, key=lambda _: _.conf))

        return new_annotations

    def predict_sample_append(self, sample: Sample) -> Sample:
        """
        Devuelve la muestra de entrada con anotaciones nuevas predichas.
        :param sample: Muestra de entrada
        :return: Muestra con nuevas anotaciones predichas.
        """
        annotations = self.predict_sample(sample)
        sorted_annotations = sorted(annotations, key=lambda _: _.conf)
        return sample.append_annotations(sorted_annotations)

    def predict_annotations_append(self, sample: Sample, r_pattern: str) -> Sample:
        """
        Devuelve la muestra de entrada con anotaciones nuevas predichas. Las
        nuevas anotaciones son predichas SOLAMENTE con respecto a las anotaciones
        previas de la muestra con la clase definida en el regex.
        :param sample: Muestra de entrada
        :param r_pattern: Filtro para las anotaciones previas.
        :return: Muestra con nuevas anotaciones predichas.
        """
        annotations = self.predict_annotations(sample, r_pattern)
        return sample.append_annotations(list_utils.flatten_list_of_lists(annotations))

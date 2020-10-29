from typing import List

import numpy as np

from Vision import Sample, Annotation
from Vision.io_managers import Manager
from Vision.models.detection_model import DetectionModel
from Vision.pipelines.pipeline import Pipeline


def keep_longest_line(digits: List[Annotation]) -> List[Annotation]:
    """
    Detects multiple lines for a list of annotations and returns the
    longest one
    :param digits: List of annotations
    :return: Longest line of annotations
    """
    if len(digits) == 0:
        return digits
    sorted_annotations = sorted(digits, key=lambda _: _.get_shape().get_left())
    prev = sorted_annotations[0]
    lines = [[prev]]
    for ann in sorted_annotations[1:]:
        x0, y0 = ann.shape.get_centroid()
        match = False
        for i, line in enumerate(lines):
            if match:
                break
            prev = line[-1]
            left, bottom, right, top = prev.shape.get_bounds()
            if bottom < y0 < top:
                lines[i].append(ann)
                match = True
        if not match:
            lines.append([ann])

    return lines[np.argmax(list(map(len, lines)))]


def fill_gaps(annotations: List[Annotation], max_pad: int = 1) -> str:
    """
    Intenta rellenar huecos entre numeros con ZEROS "0"
    :param annotations: Lista de anotacioned de numeros.
    :param max_pad: Numero maximo de ZEROS permitidos para rellenar un hueco.
    :return: Texto con los numeros y huecos rellenados con "0".
    """
    # Only perform task if annotations is not empty
    if len(annotations) == 0:
        return ""
    # Sort annotations by position from left to right
    sorted_annotations = sorted(annotations, key=lambda _: _.get_shape().get_left())
    # Allocate string container
    text = ""
    # Set first right bound
    prev_right = sorted_annotations[0].get_shape().get_left()
    approx_width = 0
    approx_overlap = sorted_annotations[0].get_shape().get_width() * 0.2
    overlap_count = 0
    for i, annotation in enumerate(sorted_annotations):
        shape = annotation.get_shape()
        # Compute size of gap
        dx = (shape.get_left() - prev_right)
        # Set previous right bound
        prev_right = shape.get_right()
        # Aproximate the average widht with the average of what has been processed so far
        approx_width = (approx_width * i + shape.get_width()) / (i + 1)
        if dx >= 0:
            # Only pad if there is a gap
            n_pad = (dx // (approx_width - 2 * approx_overlap))
        else:
            # If there is overlap, do not pad
            n_pad = 0
            # Estimate average overlap
            approx_overlap = ((approx_overlap * overlap_count) + -1 * dx) / (overlap_count + 1)
            overlap_count += 1
        text += "0" * min(max_pad, int(n_pad)) + str(annotation.cls)

    return text


def build_filter_by_min_ar(min_ar: float):
    """
    Devuelve una funcion filtro para eliminar anotaciones con un
    AR menor al indicado. AR = width/height
    :param min_ar: Minimo AR permitido.
    :return: Funcion filtro.
    """

    def check_min_ar(annotation: Annotation) -> bool:
        w = annotation.shape.get_width()
        h = annotation.shape.get_height()
        return True if (w / h > min_ar) else False

    return check_min_ar


class KmPipe(Pipeline):
    """
    Pipeline para extraer kilometraje de fotos de tablero.
    """

    def __init__(self,
                 km_detector: DetectionModel,
                 digit_detector: DetectionModel,
                 sample_manager: Manager,
                 min_len=1,
                 max_pad=1):
        super(KmPipe, self).__init__()
        self.km_detector = km_detector
        self.digit_detector = digit_detector
        self.sample_manager = sample_manager
        self.max_pad = max_pad
        self.min_len = min_len

    def get_sample_manager(self) -> Manager:
        return self.sample_manager

    def predict_sample(self, sample: Sample) -> Sample:
        km_detector = self.km_detector
        digit_detector = self.digit_detector
        min_len = self.min_len

        # First detect RoI with Km
        km_sample = km_detector.predict_sample_append(sample)
        km_roi = km_sample.filter_annotations("km")
        # Then get digits in each of the km-RoI
        digit_roi = digit_detector.predict_annotations(km_sample, "km")
        # And extract text in each RoI
        digit_text = self.__extract_digits(digit_roi)

        km_annotations = list()
        for i, ann in enumerate(km_roi):
            text = digit_text[i]
            if len(text) >= min_len:
                ann.text = digit_text[i]
                km_annotations = [ann] + digit_roi[i]

        final_sample = sample.append_annotations(km_annotations)
        return final_sample

    def __extract_digits(self, digit_roi: List[List[Annotation]]) -> List[str]:
        """
        Extrae los digitos de cada grupo de anotaciones
        :param digit_roi: Lista de listas de anotaciones.
        :return: Lista de Digitos en formato texto. Uno por cada grupo de anotaciones.
        """
        texts = list()
        for annotations in digit_roi:
            texts.append(fill_gaps(keep_longest_line(annotations), self.max_pad))
        return texts

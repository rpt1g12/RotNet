import re
from typing import Callable, Tuple

import cv2
import numpy as np

from Vision import Sample, Annotation
from Vision.shapes import Box


def create_predicate_regex_tag(pattern: str) -> Callable[[Sample], bool]:
    """
    Creates a Predicate function that returns true if the most likely tag of
    a Sample matches a certain regex pattern.
    :param pattern: Regex pattern to match
    :return: A Predicate for samples that returns true if the most likely tag of
    a Sample matches a certain regex pattern.
    """
    regex = re.compile(pattern)

    def predicate_regex_tag(sample: Sample) -> bool:
        # Get most likely tag
        tags = sample.extract_tags()
        if tags:
            tag = list(sorted(tags.items(), key=lambda t: t[1], reverse=True))[0]
            return bool(regex.match(tag[0]))
        else:
            return False

    return predicate_regex_tag


def occlude_objects(sample: Sample,
                    width_range: Tuple[float, float] = None,
                    height_range: Tuple[float, float] = None,
                    p: float = 0.5,
                    color=None,
                    r_pattern: str = None) -> Sample:
    """
    Superimpone un rectangulo sobre los objetos anotados, tapandolos/ocluyendolos parcialmente.
    :param sample: Muestra sobre la cual se desea ocluir sus objetos anotados.
    :param width_range: Tupla con el rango de valores usados para la anchura del rectangulo
    relativos al tamano del objeto a ocluir. Si `height_range` no esta especificado, la
    oclusion atraviesa la imagen entera verticalmente.
    :param height_range: Tupla con el rango de valores usados para la altura del rectangulo
    relativos al tamano del objeto a ocluir. Si `width_range` no esta especificado, la
    oclusion atraviesa la imagen entera horizontalmente.
    :param p: Probabilidad de occluir cada anotacion.
    :param color: Color por defecto en forma de tupla RGB.
    :param r_pattern: Regex para filtrar los objetos que se desean ocluir. Si es `None`,
    se ocluye la imagen.
    :return: Muestra con objetos ocluidos.
    """
    # Get shapes that will be occluded
    new_sample = sample.clone()
    # Get image numpy array
    img_arr = new_sample.get_img_arr()
    img_h, img_w = img_arr.shape[:2]
    if r_pattern is not None:
        annotations = new_sample.filter_annotations(r_pattern)
    else:
        annotations = [Annotation(Box(0, 0, img_w, img_h), "img")]

    for annotation in annotations:
        img_arr = new_sample.get_img_arr()
        # Determine if the annotation will be occluded
        occlude = np.random.rand() <= p
        if occlude:
            shape = annotation.get_shape()
            bounds = shape.get_bounds()
            w0 = shape.get_width()
            h0 = shape.get_height()
            # Choose a random color
            color = color or tuple([int(x) for x in np.random.randint(0, 101, 3, dtype=int)])
            if height_range is not None and width_range is None:
                h_factor = np.random.uniform(height_range[0], height_range[1])
                h = h0 * h_factor
                h_shift = np.random.randint(0, int(h0 - h))
                pt0 = (0, int(bounds[1] + h_shift))
                pt1 = (img_arr.shape[1], int(pt0[1] + h))
            elif width_range is not None and height_range is None:
                w_factor = np.random.uniform(width_range[0], width_range[1])
                w = w0 * w_factor
                w_shift = np.random.randint(0, int(w0 - w))
                pt0 = (int(bounds[0] + w_shift), 0)
                pt1 = (int(pt0[0] + w), img_arr.shape[0])
            elif width_range is not None and height_range is not None:
                h_factor = np.random.uniform(height_range[0], height_range[1])
                h = h0 * h_factor
                h_shift = np.random.randint(0, int(h0 - h))
                w_factor = np.random.uniform(width_range[0], width_range[1])
                w = w0 * w_factor
                w_shift = np.random.randint(0, int(w0 - w))
                pt0 = (int(bounds[0] + w_shift), int(bounds[1] + h_shift))
                pt1 = (int(pt0[0] + w), int(pt0[1] + h))
            else:
                pt0, pt1 = None, None

            if pt0 is not None and pt1 is not None:
                new_img = cv2.rectangle(img_arr, pt0, pt1, color, cv2.FILLED)
                new_sample = new_sample.substitute_img(new_img)
    return new_sample

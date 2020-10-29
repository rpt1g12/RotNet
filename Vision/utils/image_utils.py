import os
import re
from typing import Union

import numpy as np

"""
Modulo con funciones utilitarias para trabajar con imagenes.
"""


def image_path_generator(img_path):
    """
    Returns a generator for images in a directory.
    :param img_path: Directory where images will be fetched from
    :return: Files ending with image extensions
    """
    for root, dir, files in os.walk(img_path):
        for name in files:
            if is_image(name):
                yield os.path.join(root, name)


def is_image(file: str):
    """
    Checks if the file is an image of PNG, JPG or JPEG format
    :param file: File to check.
    :return: True if the file is an image of PNG, JPG or JPEG format, False otherwise
    """
    name = os.path.basename(file)
    return re.match(r"^.*\.(jpg|jpeg|png)$", name)


def return_xml_name(file: str) -> str:
    """
    Returns xml partner file for image file
    :param file: image file name
    :return: xml file name
    """
    name = re.sub(r"\.(jpg|jpeg|png)$", ".xml", file)
    return name


def crop_around_center(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def largest_rotated_rect(w: int, h: int, angle: float):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """

    quadrant = int(np.floor(angle / (np.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else np.pi - angle
    alpha = (sign_alpha % np.pi + np.pi) % np.pi

    bb_w = w * np.cos(alpha) + h * np.sin(alpha)
    bb_h = w * np.sin(alpha) + h * np.cos(alpha)

    gamma = np.arctan2(bb_w, bb_w) if (w < h) else np.arctan2(bb_w, bb_w)

    delta = np.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * np.cos(alpha)
    a = d * np.sin(alpha) / np.sin(delta)

    y = a * np.cos(gamma)
    x = y * np.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_largest_rectangle(image: np.ndarray, angle: float, height: int, width: int) -> np.ndarray:
    """
    Crop around the center the largest possible rectangle
    found with largest_rotated_rect.

    :param image: Imagen en formato array de numpy despues de la rotacion
    :param angle: Angulo que se ha rotado la imagen.
    :param height: Alto de la imagen previo a la rotacion.
    :param width: Ancho de la imagen previo a la rotacion.
    """
    return crop_around_center(
        image,
        *largest_rotated_rect(
            width,
            height,
            float(np.radians(angle))
        )
    )


def get_affine_matrix(center: tuple = (0.0, 0.0), angle: float = 0.0, scale: Union[tuple, float] = 1.0) -> np.ndarray:
    """
    Calculates an affine transformation matrix that rotates arround a center and scales.

    The transformation follows the order:
        1. Scaling.
        2. Translation so that center is new origin of coordinates.
        3. Rotation.
        4. Translation back to original position.

    @param center Center of the rotation in the source image.
    @param angle Rotation angle in degrees. Positive values mean counter-clockwise rotation (the
    coordinate origin is assumed to be the top-left corner).
    @param scale Scale factor in both directions. If a float, use same scaling in both directions.
    """
    if not isinstance(scale, tuple):
        scale = (scale, scale)

    # Angle in radians
    theta = np.radians(angle)
    # Cosine and sine
    cos = np.cos(theta)
    sin = np.sin(theta)

    # Transformation matrices
    rot_M = np.array([
        [cos, sin, 0],
        [-sin, cos, 0],
        [0, 0, 1],
    ])
    scl_M = np.array([
        [scale[0], 0, 0],
        [0, scale[1], 0],
        [0, 0, 1],
    ])
    trans_0_M = np.array([
        [1, 0, -center[0] * scale[0]],
        [0, 1, -center[1] * scale[1]],
        [0, 0, 1],
    ])
    trans_1_M = np.array([
        [1, 0, center[0] * scale[0]],
        [0, 1, center[1] * scale[1]],
        [0, 0, 1],
    ])

    M = np.dot(trans_1_M,
               np.dot(
                   rot_M,
                   np.dot(trans_0_M, scl_M)
               ))[:2, :]

    return M

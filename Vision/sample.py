import copy
from itertools import cycle
from typing import Generator, List, Union, Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from matplotlib import colors as plt_colors
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from Vision.annotation import Annotation_List, class_filter
from Vision.shapes import Box
from Vision.utils import DefaultDictAddMissing, image_utils

# Constants
COLOR_CYCLE = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
CMAP = plt_colors.LinearSegmentedColormap.from_list("", ["red", "yellow", "green"])


class Sample(object):
    class_color_dict = DefaultDictAddMissing(lambda c: (next(COLOR_CYCLE)))
    """
    Clase que contiene un ejemplo para entrenar
    """

    def __init__(self, sample_id: int, name: str, img: Image, annotations=None,
                 segmented: bool = False, metadata=None):
        """
        Constructor de la clase a partir de un Id, un nombre de archivo de imagen,
        un objeto Image de PIL y una lista de objetos Annotation
        :param sample_id: Id del ejemplo.
        :param name: Nombre del archivo que contiene la imagen.
        :param img: Objeto Image de PIL con la imagen del ejemplo.
        :param annotations: Lista de objetos Annotation con las anotaciones
        :param segmented: Indica si es un objeto segmentado o no.
        :param metadata: Metadatos de la muestra.
        """
        super(Sample, self).__init__()
        if metadata is None:
            metadata = {}
        if annotations is None:
            annotations = []
        self.sample_id = sample_id
        self.name = name
        self.img = img
        self.annotations = annotations
        self.segmented = segmented
        self.metadata = metadata

    def plot(self, ax: Axes = None, with_confidence=True, with_text=False,
             cls_filt: str = r".*", *args, **kwargs) -> None:
        """
        Pinta un ejemplo en Matplotlib con todas sus anotaciones.
        :param ax: Objeto Axes de Matplotlib donde pintar el ejemplo.
        :param with_text: True si queremos anadir texto adicional de la anotacion
        :param with_confidence: True si queremos anadir la confianza de la anotacion.
        :param cls_filt: Filtro para las clases en formato regex.
        """
        if ax is None:
            fig, ax = plt.subplots()
            fig.suptitle("{}: {}".format(self.sample_id, self.name))
        ax.imshow(self.img)
        # Plot detections
        for annotation in self.filter_annotations(cls_filt):
            color = self.get_class_color(annotation.cls)
            annotation.add_to_plot(ax=ax, color=color,
                                   with_confidence=with_confidence, with_text=with_text)
        # Add tags in a legend
        if self.extract_tags():
            self.create_tag_legend(ax)

        pass

    def create_tag_legend(self, ax: Axes) -> None:
        """
        Creates a legend with the tags contained in the metadata.
        :param ax: Axis to add the legend to
        """
        legend_elements = list()
        tags = self.extract_tags()

        for k, v in tags.items():
            legend_elements.append(
                Line2D(
                    [0], [0], marker="o", lw=0, markersize=8, color=CMAP(v), label=f"{k}:{v:.3f}"
                )
            )
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    def crop(self, cls_filt: str = r".*") -> List[Image.Image]:
        """
        Devuelve las anotaciones recortadas de la imagen.
        :param cls_filt: Filtro para las clases en formato regex.
        :return: Lista de imagenes de las regiones recortadas.
        """
        crops = list()
        for annotation in self.filter_annotations(cls_filt):
            shape = annotation.get_shape()
            crops.append(self.img.crop(shape.get_bounds()))
        return crops

    def zoom_to_annotations(self, grow_w: float = 1.2, grow_h: float = 1.2,
                            cls_filt: str = r".*") -> 'Sample':
        """
        Devuelve la muestra con zoom a las anotaciones.
        :param grow_w: Factor para incluir padding sobre las anotaciones en horizontal.
        :param grow_h: Factor para incluir padding sobre las anotaciones en vertical.
        :param cls_filt: Filtro para las anotaciones.
        :return: Muestra con zoom sobre las anotaciones. Si no existen anotaciones, o
        las anotaciones de la muestra no pasan el filtro, se devuelve la muestra
        original sin anotaciones.
        """
        filtered_annotations = self.filter_annotations(cls_filt)
        n_annotations = len(filtered_annotations)
        if n_annotations == 0:
            new_sample = self.clone()
            new_sample.annotations = list()
            return new_sample

        bound_array = np.zeros((4, n_annotations))
        for i, annotation in enumerate(filtered_annotations):
            shape = annotation.get_shape()
            bounds = shape.get_bounds()
            bound_array[:, i] = np.array(bounds)

        minimum = np.min(bound_array, axis=1, keepdims=True)[:2]
        x_min, y_min = minimum[0, 0], minimum[1, 0]
        maximum = np.max(bound_array, axis=1, keepdims=True)[2:]
        x_max, y_max = maximum[0, 0], maximum[1, 0]
        if grow_w > 1:
            w = x_max - x_min
            x_min = int(x_max - 0.5 * w * (1 + grow_w))
            x_max = int(np.ceil(x_min + w * grow_w))
        if grow_h > 1:
            h = y_max - y_min
            y_min = int(y_max - 0.5 * h * (1 + grow_h))
            y_max = int(np.ceil(y_min + h * grow_h))

        img = self.get_img()
        max_x = img.width
        max_y = img.height
        new_bounds = (
            max(x_min, 0),
            max(y_min, 0),
            min(x_max, max_x) + 1,
            min(y_max, max_y) + 1
        )
        new_img = img.crop(new_bounds)

        new_annotations = list()
        for annotation in filtered_annotations:
            ann = annotation.clone()
            shape = annotation.get_shape().clone()
            shape.shift_origin(-new_bounds[0], -new_bounds[1])
            ann.shape = shape
            new_annotations.append(ann)

        return Sample(self.sample_id, self.name, new_img, new_annotations, self.segmented, self.metadata)

    def is_segmented(self):
        return self.segmented

    @staticmethod
    def get_class_color(cls):
        """
        Devuelve siempre el mismo color para una clase.
        :param cls: Clase para la que devolver un color
        :return: Color en formato hexadecimal
        """
        return Sample.class_color_dict[cls]

    def filter_annotations(self, r_pattern) -> Annotation_List:
        """
        Devuelve una lista de anotaciones cuya clase pasa el filtro del regex.
        :param r_pattern: Regex para filtrar las clases.
        :return: Lista de anotaciones.
        """
        predicate = class_filter(r_pattern)
        return list(filter(predicate, self.annotations))

    def get_img(self) -> Image:
        """
        Devuelve la imagen como objeto de PIL
        :return: Imagen como objeto PIL.Image
        """
        return self.img

    def get_img_arr(self) -> np.ndarray:
        """
        Devuelve la imagen como un array de numpy
        :return: Array de numpy con los datos de la imagen.
        """
        img_array = np.array(self.get_img())
        # Check if image is in grayscale
        if len(img_array.shape) < 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        return img_array

    def get_metadata(self) -> dict:
        """
        Devuelve los metadatos de la muestra
        :return: Metadatos de la muestra.
        """
        return self.metadata

    def append_annotations(self, annotations: Annotation_List) -> 'Sample':
        """
        Devuelve una muestra nueva con las anotaciones adicionales.
        :param annotations: Anotaciones adicionales.
        :return: Nueva muestra con anotaciones adicionales.
        """
        new_sample = self.clone()
        new_sample.annotations += annotations
        return new_sample

    def apply_rotation(self, crop_inner_rect=False) -> 'Sample':
        """
        Aplica la rotacion dada por la clave: `'theta'` contenida en el campo `'metadata'`
        de la muestra. La rotacion
        :param crop_inner_rect: Se recorta el rectangulo con maxima area contenido
        en la imagen girada.
        :return: Muestra rotada por el angulo `'theta'`
        """
        # Obtain original witdth and height
        h, w = self.get_img_arr().shape[:2]
        angle = self.get_rotation() % 360
        angle = 360 - angle
        # Rotate the image
        rotated_sample = self.affine(-self.get_rotation(), borderMode=None)
        # Zero rotation
        self.set_rotation(0)
        if crop_inner_rect:
            # Extract maximum inner rectangle
            rotated_sample.set_img(image_utils.crop_largest_rectangle(rotated_sample.get_img_arr(), angle, h, w))
            return rotated_sample

        return rotated_sample

    def affine(self, angle: float, scale: Tuple[float, float] = (1.0, 1.0),
               borderMode=cv2.BORDER_REPLICATE) -> 'Sample':
        """
        Aplica una transformacion afin compuesta por:
            * Una rotacion en el sentido opuesto a las agujas del reloj
            * Un escalado
        :param angle: Angulo en grados a rotar en el sentido de las agujas del reloj.
        :param scale: Tupla de factores de escalado de los ejes horizontal y vertical
        :param borderMode: Tipo de relleno usado para el borde en caso de rotacion.
        Por ejemplo, `cv2.BORDER_TRANSPARENT` o `cv2.BORDER_REPLICATE`
        :return: Muestra transformada
        """
        # Affine transformation matrix and the new size of the rotated image
        M, (n_w, n_h) = self.get_rotation_matrix_and_size(angle, scale)
        # Apply affine transform
        if borderMode:
            img_arr = cv2.warpAffine(
                self.get_img_arr(),
                M,
                (n_w, n_h),
                flags=cv2.INTER_LINEAR,
                borderMode=borderMode,
            )
        else:
            img_arr = cv2.warpAffine(
                self.get_img_arr(),
                M,
                (n_w, n_h),
                flags=cv2.INTER_LINEAR
            )
        self.set_img(img_arr)
        for annotation in self.annotations:
            shape = annotation.get_shape()
            shape.affine(M)
        return self

    def resize(self, img_shape) -> 'Sample':
        # Get current img_shape
        h, w = self.get_img_arr().shape[:2]
        # Reshape image
        self.img = Image.fromarray(cv2.resize(self.get_img_arr(), img_shape))

        scale = (img_shape[0] / w, img_shape[1] / h)

        # Reshape annotations
        for annotation in self.annotations:
            shape = annotation.get_shape()
            shape.resize(scale)

        return self

    def get_rotation_matrix_and_size(self, angle, scale=(1.0, 1.0)):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = self.get_img_arr().shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = image_utils.get_affine_matrix((cX, cY), angle, scale)
        box = Box(0, 0, w, h)
        box.affine(M)
        bounds = box.get_bounds()
        n_w = bounds[2] - bounds[0]
        n_h = bounds[3] - bounds[1]
        M[0, 2] += ((n_w / 2) - cX * scale[0])
        M[1, 2] += ((n_h / 2) - cY * scale[1])
        return M, (n_w, n_h)

    def clone(self) -> 'Sample':
        return copy.deepcopy(self)

    def set_img(self, img: Union[np.ndarray, Image.Image]) -> 'Sample':
        if isinstance(img, Image.Image):
            self.img = img
        else:
            self.img = Image.fromarray(img)
        return self

    def substitute_img(self, img: Union[np.ndarray, Image.Image]) -> 'Sample':
        new_sample = self.clone()
        new_sample.set_img(img)
        return new_sample

    def substitute_annotations(self, annotations: Annotation_List) -> 'Sample':
        new_sample = self.clone()
        new_sample.annotations = annotations
        return new_sample

    def set_metadata(self, metadata) -> 'Sample':
        self.metadata = metadata
        return self

    def extract_tags(self) -> Dict[str, float]:
        return self.get_metadata().get("tags")

    def get_rotation(self) -> float:
        theta = self.get_metadata().get("theta")
        return theta if theta is not None else 0.0

    def set_rotation(self, theta: float) -> 'Sample':
        metadata = self.get_metadata()
        metadata["theta"] = theta
        self.set_metadata(metadata)
        return self


# Define derived types
Sample_List = List[Sample]
Sample_Generator = Generator[Sample, None, None]

import os
import xml.etree.ElementTree as ET
from _elementtree import Element
from ast import literal_eval
from itertools import groupby
from typing import List, Dict

import numpy as np
from PIL import Image
from jinja2 import Environment, PackageLoader

from Vision import Annotation
from Vision.io_managers import __name__ as PACKAGE_NAME
from Vision.io_managers.manager import Manager
from Vision.sample import Sample, Sample_Generator, Sample_List
from Vision.shapes import Box
# CONSTANTS
from Vision.shapes.polygon import Polygon
from Vision.utils import image_utils
from Vision.utils.parallelisation import threadsafe_generator

VOC_TEMPLATE = Environment(
    loader=PackageLoader(PACKAGE_NAME)
).get_template("voc_template.xml")


def include_tags_in_object(obj: dict, tags: Dict[str, float]) -> dict:
    new_obj = obj.copy()
    new_obj["levels"] = [dict(name=k, value=f"{v:.4f}") for k, v in tags.items()]
    return new_obj


class VOC(Manager):
    """
    Clase para manejar ficheros en formato Pascal VOC
    """

    def __init__(self,
                 in_path: str = None,
                 out_path: str = None,
                 source: str = "Unknown",
                 batch_size=32,
                 default_tags=None,
                 shuffle=False,
                 seed=0
                 ):
        """
        Constructor de la clase
        :param in_path: Path desde donde se leen las muestras
        :param out_path: Path donde se guardan las muestras
        :param source: Fuente de los datos
        :param default_tags: Lista de tags por defecto usadas en
        clasificion. Estas tags se añaden a las que ya vengan incluidas
        en la muestra via su fichero VOC correspondiente.
        """
        self.batch_size = batch_size
        self.in_path = in_path
        self.out_path = out_path
        self.source = source
        if in_path is not None:
            self.list_dir = os.listdir(self.in_path)
        else:
            self.list_dir = []
        self.file_list = list(filter(image_utils.is_image, self.list_dir)) if in_path else []
        # Keep only images with xml pair
        self.file_list = [file for file in self.file_list
                          if image_utils.return_xml_name(file) in self.list_dir]
        if default_tags is None:
            default_tags = []
        self.default_tags = default_tags

        # Initialise Keras Iterator
        super(VOC, self).__init__(in_path, out_path, self.file_list,
                                  batch_size, shuffle, seed)

    def clone(self) -> 'VOC':
        return VOC(in_path=self.in_path,
                   out_path=self.out_path,
                   source=self.source,
                   batch_size=self.batch_size,
                   default_tags=self.default_tags,
                   shuffle=self.shuffle,
                   seed=self.seed)

    def set_in_path(self, path):
        self.in_path = path
        self.set_file_list(list(filter(image_utils.is_image, os.listdir(self.in_path))) if path else [])

    def read_sample(self, n) -> Sample:
        if n < 0:
            n = np.random.randint(0, len(self.file_list))
        assert len(self.file_list) > n, f"No existen tantas muestras en {self.in_path}"
        img_name = self.file_list[n]
        voc_file = os.path.join(self.in_path, f"{os.path.splitext(img_name)[0]}.xml")
        sample = self.__to_sample(n, voc_file)
        return sample

    def write_sample(self, sample: Sample, write_image=True) -> int:
        assert self.out_path is not None, "No se ha especificado un path donde guardar los ficheros VOC"
        try:
            voc = self.to_voc_dict(sample)
        except Exception as e:
            raise Exception("Error en muestra %s" % sample.name)
        out_file = f"{os.path.splitext(voc['path'])[0]}.xml"
        with open(out_file, "w+") as fh:
            fh.write(VOC_TEMPLATE.render(**voc))
        if write_image:
            sample.img.save(voc["path"])
        return 1

    @threadsafe_generator
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        sample_id = 0
        samples = list()
        for file in self.__file_generator():
            sample = self.__to_sample(sample_id, file)
            samples.append(sample)
            sample_id += 1
            if len(samples) % self.batch_size == 0:
                out_samples = samples
                samples = list()
                yield out_samples
        yield samples

    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        count = 0
        for sample in samples:
            count += self.write_sample(sample, write_image)
        return count

    def to_voc_dict(self, sample: Sample) -> dict:
        """
        Convierte un objeto Sample a un diccionario con los campos de un fichero VOC.
        :param sample: Muestra a convertir
        :return: Diccionario con los campos de un fichero VOC
        """
        assert self.out_path is not None, "No se ha especificado un path donde guardar los ficheros VOC"
        voc = dict()
        voc["folder"] = self.out_path
        voc["filename"] = sample.name
        voc["path"] = os.path.join(voc["folder"], voc["filename"])
        voc["source"] = dict(database=self.source)
        # Get image shape/size
        img_array = sample.get_img_arr()
        voc["size"] = dict(
            height=img_array.shape[0],
            width=img_array.shape[1],
            depth=img_array.shape[2]
        )
        voc["segmented"] = int(sample.is_segmented())
        voc["rotation"] = int(sample.get_rotation())
        voc["objects"] = self.get_objects(sample)

        return voc

    def get_objects(self, sample: Sample) -> List[dict]:
        """
        Extrae las anotaciones de una muestra y los convierte a una lista
        de diccionarios compatible con la etiqueta 'object' de un fichero
        VOC.
        :param sample: Muestra de donde extraer las anotaciones
        :return: Lista de diccionarios en formato 'object'
        """
        objects = list()
        # Loop through all annotations in the sample
        for annotation in sample.annotations:
            shape = annotation.get_shape()
            # Extract data common to all kind of shapes
            obj = dict(
                name=annotation.get_cls(),
                pose="Unspecified",
                truncated=0,
                occluded=0,
                difficult=0,
                confidence=annotation.conf
            )
            # Extract specific data depending of type of shape
            if type(shape) == Box:
                obj["bndbox"] = dict(
                    xmin=shape.get_left(),
                    ymin=shape.get_bottom(),
                    xmax=shape.get_right(),
                    ymax=shape.get_top()
                )
            elif type(shape) == Polygon:
                obj["polygon"] = shape.vertices
            objects.append(obj)
        # Extract tags if any
        tags = sample.extract_tags()
        # And include them as an object
        if tags:
            # Include default tags just in case they are not present
            # already with a default value of 0.0
            default_tags = {t: 0.0 for t in self.default_tags}
            default_tags.update(tags)
            obj = dict(
                name=os.path.basename(os.path.normpath(self.out_path)),
                pose="Unspecified",
                truncated=0,
                occluded=0,
                difficult=0,
            )
            obj = include_tags_in_object(obj, default_tags)
            objects.append(obj)
        return objects

    @staticmethod
    def parse_voc(file_path) -> dict:
        """
        Extrae los campos de un fichero VOC y los pasa a un diccionario.
        :param file_path: Ruta al fichero VOC.
        :return: Diccionario con los campos del fichero VOC.
        """
        voc = dict()
        root = ET.parse(file_path).getroot()
        voc = VOC.__parse_xml(voc, root)
        # Make sure metadata is not None
        voc["metadata"] = dict()
        # Make sure objects is a list
        if 'object' in voc.keys():
            voc['objects'] = [voc.pop('object')]

        return voc

    @staticmethod
    def __count_sibilings(root: Element) -> Dict[str, int]:
        """
        Counts how many xml tags of every kind are contained in the root tag
        :param root: XML root element.
        :return: Dictionary with child-element-kind's count.
        """
        group_dicts = {
            k: sum(1 for _ in g) for k, g in
            groupby(
                root.getchildren(),
                lambda c: c.tag
            )
        }
        return group_dicts

    @staticmethod
    def __parse_xml(container: dict, root: Element) -> dict:
        """
        Extrae todos los descendientes de un nodo XML y los guarda en un
        diccionario. Si existen varios descendientes con la misma etiqueta
        los agrupa en una lista.
        :param container: Diccionario donde se guardan los nodos descendientes.
        :param root: Nodo raiz de donde extraer los descendientes.
        :return: Diccionario actualizado con los descendientes extraidos.
        """
        child_counts = VOC.__count_sibilings(root)
        for child in root.getchildren():
            child_tag = child.tag
            # Find if how many of the children are
            if child_counts[child_tag] > 1:  # We make a list of the objects
                brothers_tag = f"{child_tag}s"
                brothers = container.get(brothers_tag)
                if brothers:
                    container[brothers_tag].append(VOC.__parse_xml(dict(), child))
                else:
                    container[brothers_tag] = [VOC.__parse_xml(dict(), child)]
            else:  # We check if they have children too
                if len(child.getchildren()) > 0:
                    container[child.tag] = VOC.__parse_xml(dict(), child)
                else:  # If they don't we parse the inner text value
                    try:  # Let Python parse the value and assign its type
                        val = literal_eval(child.text)
                    except ValueError as e:  # Its a plain string
                        val = child.text
                    except SyntaxError as e:  # Contains the \ character
                        val = child.text
                    container[child.tag] = val
        return container

    def __file_generator(self):
        assert self.in_path is not None, "No se ha especificado un path de donde leer los ficheros VOC"
        for file in self.file_list:
            name = image_utils.return_xml_name(file)
            yield os.path.join(self.in_path, name)

    def __to_sample(self, sample_id, voc_file):
        voc = VOC.parse_voc(voc_file)
        annotations = list()
        metadata = dict()

        # Parse rotation angle of image if available
        metadata["theta"] = voc.get("rotation") or 0

        # Parse annotations/classification if available
        objects = voc.get("objects") or []
        for obj in objects:
            metadata.update(self.extract_metadata_from_object(obj))
            cls = str(obj["name"])
            conf = obj.get("confidence")
            if not conf:
                conf = 1.0
            bndbox = obj.get("bndbox")
            polygon = obj.get("polygon")
            if bndbox:
                shape = Box.from_minmax(
                    obj["bndbox"]["xmin"],
                    obj["bndbox"]["ymin"],
                    obj["bndbox"]["xmax"],
                    obj["bndbox"]["ymax"]
                )
                annotations.append(Annotation(shape, cls, conf=conf))
            if polygon:
                vertices = list(map(lambda p: (p["x"], p["y"]), polygon["points"]))
                shape = Polygon(vertices)
                annotations.append(Annotation(shape, cls, conf=conf))
        img = Image.open(os.path.join(self.in_path, voc["filename"]))
        sample = Sample(sample_id, voc["filename"], img, annotations, metadata=metadata)
        return sample

    def extract_metadata_from_object(self, obj: dict) -> dict:
        """
        Extracts tags from an object dictionary parsed from a VOC file.
        :param obj: Object dictionary with parsed data.
        :return: Dictionary with metadata attached to the sample. Possibly,
        tags from a classification task.
        """
        metadata = dict()
        levels = list()
        # Check if dict contains classification or multi-classification tasks
        if obj.get("classification"):
            levels = obj.get("classification").get("levels")
        elif obj.get("multiclassification"):
            levels = obj.get("multiclassification").get("levels")
        # If it does, extract the labels. Do it also if default tags are set.
        if len(levels) or len(self.default_tags):
            tags = {t: 0.0 for t in self.default_tags}
            new_tags = {level["name"]: float(level["value"]) for level in levels}
            tags.update(new_tags)
            metadata["tags"] = tags

        return metadata

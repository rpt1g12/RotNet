import os
from functools import reduce

import h5py
import numpy as np
from PIL import Image

from Vision import Sample
from Vision.annotation import Annotation, Annotation_List
from Vision.io_managers.manager import Manager
from Vision.shapes import Box
from Vision.utils.parallelisation import threadsafe_generator


class SVHN(Manager):
    """
    Clase que se encarga de cargar ficheros .mat procedentes
    del dataset SVHN
    """

    def __init__(self, path, batch_size=32, shuffle=False, seed=0):
        """
        Constructor a partir del path del directorio donde se encuentran las
        imagenes y el archivo HDF5 con las anotaciones.
        :param path:
        """
        self.batch_size = batch_size
        self.path = path
        self.hdf5 = self.__load_hdf5()
        self.digitStruct = self.hdf5["digitStruct"]
        self.bbox = self.digitStruct["bbox"]
        self.name = self.digitStruct["name"]
        super(SVHN, self).__init__(path, None, [i for i in range(self.name.shape[0])],
                                   batch_size, shuffle, seed)

    def clone(self) -> 'SVHN':
        return SVHN(
            path=self.path,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed
        )

    def _get_name(self, n: int) -> str:
        """
        Devuelve el nombre de la imagen correspondiente al enesimo ejemplo del
        dataset.
        :param n: Id del ejemplo.
        :return: Nombre de la imagen del ejemplo.
        """
        name = reduce(lambda x, y: x + chr(y), np.squeeze(self.hdf5[self.name[n, 0]]), '')
        return name

    def __get_box_attribute(self, container: dict, key, val) -> None:
        """
        Se encarga de extraer los atributos correspondientes a las cajas
        anotadas de un ejemplo y guarda los resultados en un diccionario
        contenedor.
        :param container: Diccionario donde se guardan los atributos
        extraidos
        :param key: Clave del atributo extraido
        :param val: Valor del atributo extraido. En el caso de que el ejemplo
        solo contenga una caja este parametro ya contiene el valor del atributo.
        En caso contrario, este parametro contiene las referencias a los valores
        del atributo por cada caja.
        """
        if val.shape[0] == 1:
            container[key] = val[0]
        else:
            container[key] = np.array([self.hdf5[ref[0]][0, 0] for ref in val])

    def _get_annotations(self, n: int) -> Annotation_List:
        """
        Devuelve una lista con annotaciones para el enesimo ejemplo del
        dataset.
        :param n: Id del ejemplo.
        :return: Lista de objetos Annotation.
        """
        box_data = self._get_bboxes(n)
        annotations = list()
        for data in box_data:
            box = Box(data["left"], data["top"], data["width"], data["height"])
            annotations.append(Annotation(box, str(data["label"])))
        return annotations

    def _get_sample(self, n):
        """
        Devuelve el enesimo ejemplo del dataset.
        :param n: Id del ejemplo.
        :return: Objeto Ejemplo.
        """
        annotations = self._get_annotations(n)
        name = self._get_name(n)
        img = Image.open(os.path.join(self.path, name))
        fp = img.fp
        img.load()
        # TODO: is the following statement necessary?
        fp.closed
        return Sample(n, name, img, annotations)

    def _get_bboxes(self, n):
        """
        Devuelve una lista con los atributos de las cajas contenidas en el
        enesimo ejemplo.
        :param n: Id del ejemplo.
        :return:  Lista de diccionarios con los atributos de las cajas.
        """
        item = self.hdf5[self.bbox[n, 0]]
        bboxes = {}
        item.visititems(lambda k, v: self.__get_box_attribute(bboxes, k, v))

        # Check number of boxes extracted
        n_boxes = bboxes[list(bboxes.keys())[0]].shape[0]
        box_list = list()

        # Rearrange into a list of boxes
        for i in range(n_boxes):
            box = dict()
            for k, v in bboxes.items():
                box[k] = v[i]
            box_list.append(box)
        return box_list

    def __load_hdf5(self):
        """
        Carga el fichero en formato HDF5 de SVHN.
        :return: Objeto HDF5.File
        """
        mat_file = os.path.join(self.path, "digitStruct.mat")
        obj = h5py.File(mat_file, "r")
        return obj

    def read_sample(self, n) -> Sample:
        return self._get_sample(n)

    def write_sample(self, sample: Sample, **kwargs) -> int:
        print("This manager does not write samples.")
        return 0

    @threadsafe_generator
    def sample_generator(self, batch_size: int=0):
        if batch_size>0:
            self.set_batch_size(batch_size)
        n_samples = self.name.shape[0]
        samples = list()
        for i in range(n_samples):
            samples.append(self.read_sample(i))
            if (len(samples) % self.batch_size == 0) or (i == n_samples - 1):
                out_samples = samples
                samples = list()
                yield out_samples

    def write_samples(self, samples: list, **kwargs) -> int:
        return 0

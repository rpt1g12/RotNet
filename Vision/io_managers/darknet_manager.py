import os
import re
from ast import literal_eval

import numpy as np
from PIL import Image
from jinja2 import Environment, PackageLoader

from Vision import Annotation
from Vision.io_managers import Manager
from Vision.io_managers import __name__ as PACKAGE_NAME
from Vision.sample import Sample, Sample_List, Sample_Generator
from Vision.shapes import Box
from Vision.utils import image_utils
from Vision.utils.dict_utils import invert_map
# CONSTANTS
from Vision.utils.parallelisation import threadsafe_generator

DARKNET_TEMPLATE = Environment(
    loader=PackageLoader(PACKAGE_NAME)
).get_template("darknet_template.txt")


class DarknetManager(Manager):
    """
    Clase para manejar ficheros en formato Darknet.
    """

    def __init__(self, class_map: dict, in_path: str = None, out_path: str = None,
                 obj_dir_name="obj", separator=" ",
                 batch_size=32, shuffle=False, seed=0):
        self.batch_size = batch_size
        self.in_path = in_path
        self.out_path = out_path
        self.obj_dir_name = obj_dir_name
        self.out_obj_dir = self.__create_obj_dir(out_path) if out_path else None
        self.in_obj_dir = self.__create_obj_dir(in_path) if in_path else None
        self.file_list = list(filter(image_utils.is_image, os.listdir(self.in_obj_dir))) if in_path else []
        self.class_map = class_map
        self.inv_class_map = invert_map(class_map)
        self.separator = separator
        super(DarknetManager, self).__init__(in_path, out_path, self.file_list,
                                             batch_size, shuffle, seed)

    def clone(self) -> 'DarknetManager':
        return DarknetManager(
            class_map=self.class_map,
            in_path=self.in_path,
            out_path=self.out_path,
            obj_dir_name=self.obj_dir_name,
            separator=self.separator,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed
        )

    def set_separator(self, separator: str) -> None:
        self.separator = separator

    def set_in_path(self, in_path):
        self.in_path = in_path
        self.in_obj_dir = self.__create_obj_dir(in_path) if in_path else None
        self.set_file_list(list(filter(image_utils.is_image, os.listdir(self.in_obj_dir))) if in_path else [])

    def set_out_path(self, out_path):
        self.out_path = out_path
        self.out_obj_dir = self.__create_obj_dir(out_path) if out_path else None

    @staticmethod
    def load_class_file(class_file: str) -> dict:
        """
        Carga un fichero de clases en formato DARKNET. Un nombre de clase por cada linea
        del fichero.
        :param class_file: Ruta al fichero de clases.
        :return: Diccionario con nombres de clases y ids de clase
        """
        class_map = dict()
        with open(class_file, "r") as fh:
            for line_num, line in enumerate(fh):
                class_map[line.strip()] = line_num
        return class_map

    @classmethod
    def from_class_file(cls, class_file: str):
        """
        Inicializa la clase a partir de un fichero de clases DARKNET. Utiliza
        el directorio donde se encuentre este como directorio I/O
        :param class_file: Ruta al fichero de clases.
        :return: Instancia de la clase
        """
        in_path = os.path.dirname(class_file)
        return cls(cls.load_class_file(class_file), in_path, in_path)

    def read_sample(self, n) -> Sample:
        if n < 0:
            n = np.random.randint(0, len(self.file_list))
        assert len(self.file_list) > n, f"No existen tantas muestras en {self.in_path}"
        img_name = self.file_list[n]
        file = os.path.join(self.in_obj_dir, img_name)
        return self.__to_sample(n, file)

    def write_sample(self, sample: Sample, write_image=False) -> int:
        assert self.out_path is not None, "No se ha especificado un path donde guardar los ficheros DARKNET"

        darknet = self.to_darknet_dict(sample)
        out_file = os.path.join(
            self.out_obj_dir,
            f"{os.path.splitext(sample.name)[0]}.txt"
        )
        with open(out_file, "w") as fh:
            fh.write(DARKNET_TEMPLATE.render(**darknet))
        if write_image:
            out_image = os.path.join(self.out_obj_dir, sample.name)
            sample.img.save(out_image)
        return 1

    @threadsafe_generator
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        assert self.in_path is not None, "No se ha especificado un path de donde leer los ficheros DARKNET"
        sample_id = 0
        samples = list()
        for file in self.__file_generator():
            samples.append(self.__to_sample(sample_id, file))
            sample_id += 1
            if len(samples) % self.batch_size == 0:
                out_samples = samples
                samples = list()
                yield out_samples
        yield samples

    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        count = 0
        for sample in samples:
            count += self.write_sample(sample, write_image=write_image)
        return count

    def __file_generator(self):
        """
        Generador de ficheros en formato JPG, JPEG o PNG dentro del directorio obj de entrada.
        :return: Generador de ficheros.
        """
        assert self.in_path is not None, "No se ha especificado un path de donde leer los ficheros DARKNET"
        for root, dir, files in os.walk(self.in_obj_dir):
            for name in files:
                if re.match(r"^.*\.(jpg|jpeg|png)$", name):
                    yield os.path.join(root, name)

    def to_darknet_dict(self, sample):
        """
        Convierte una Muestra en un diccionario para ser utilizado en el template.
        :param sample: Muestra a ser convertida.
        :return: Diccionario preparado para usar en el template.
        """
        # Get image width and height
        img_w = sample.img.width
        img_h = sample.img.height
        # Initialise dictionary
        darknet = dict(annotations=list())
        for annotation in sample.annotations:
            shape = annotation.shape
            centroid = shape.get_centroid()
            bounds = shape.get_bounds()
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            cls = annotation.cls
            if cls not in self.class_map.keys():
                continue  # Skip annotation if not on class-map
            # Coordinates and sizes relative to image size
            ann = dict(
                cls=self.class_map[cls],
                x=centroid[0] / img_w,
                y=centroid[1] / img_h,
                w=width / img_w,
                h=height / img_h
            )
            darknet["annotations"].append(ann)
        return darknet

    def __to_sample(self, sample_id: int, img_path: str) -> Sample:
        """
        Extra un objeto muestra de un fichero de anotacion en formato DARKNET
        :param sample_id: Id de la muestra
        :param img_path: Path a la imagen de la muestra. El fichero .txt usara el mismo
        nombre pero con extension .txt.
        :return: Objeto muestra extraido.
        """
        # Load image
        img = Image.open(img_path)
        fp = img.fp
        img.load()
        # TODO: Is the following statement necessary?
        fp.closed
        # Get image width and height
        img_w = img.width
        img_h = img.height
        # Get path to annotations file
        obj_path = f"{os.path.splitext(img_path)[0]}.txt"
        annotations = list()
        separator = self.separator
        with open(obj_path, "r") as fh:
            for line in fh:
                vals = line.strip().split(separator)
                if len(vals):
                    cls = self.inv_class_map[literal_eval(vals[0])]
                    x = literal_eval(vals[1]) * img_w
                    y = literal_eval(vals[2]) * img_h
                    w = literal_eval(vals[3]) * img_w
                    h = literal_eval(vals[4]) * img_h
                    if len(vals) > 5:
                        conf = float(vals[5])
                    else:
                        conf = 1
                    if int(h * w) == 0:
                        continue
                    annotations.append(Annotation(Box.from_centdim(x, y, w, h), cls=cls, conf=conf))
        name = os.path.basename(img_path)
        sample = Sample(sample_id, name, img, annotations)

        return sample

    def write_obj_data(self, train_split=0.8, seed=0) -> None:
        """
        Writes image all files needed by darknet to train a model:
            * obj.data : contains info about how many classes there are,
            lists of files for train, test and validation, and the file
            where the classes are.
        :param train_split: percentage of the data used for training. Test and
        validation split in half the remaining part.
        :param seed: Seed for the random split.
        """
        # Write main data file
        with open(os.path.join(self.out_path, "obj.data"), "w") as fh:
            fh.write(f"classes = {len(self.class_map)}\n")
            fh.write(f"train = data/train.txt\n")
            fh.write(f"test = data/test.txt\n")
            fh.write(f"valid = data/valid.txt\n")
            fh.write(f"names = data/obj.names\n")
            fh.write(f"backup = backup/")

        # Write names file
        with open(os.path.join(self.out_path, "obj.names"), "w") as fh:
            class_ids = list(self.inv_class_map.keys())
            class_ids.sort()
            for key in class_ids:
                fh.write(f"{self.inv_class_map[key]}\n")

        # Get a list of all samples
        img_files = list()
        for file in self.__file_generator():
            img_files.append(file)

        # Random shuffle the list of samples
        np.random.seed(seed=seed)
        np.random.shuffle(img_files)

        # Split train and test datasets
        n_train = int(len(img_files) * train_split)
        img_train = img_files[:n_train]
        n_test_valid = (len(img_files) - n_train) // 2
        img_test = img_files[n_train:n_train + n_test_valid]
        img_valid = img_files[n_train + n_test_valid:]

        # Write train file
        with open(os.path.join(self.out_path, "train.txt"), "w") as fh:
            fh.write("\n".join(img_train))
        # Write test file
        with open(os.path.join(self.out_path, "test.txt"), "w") as fh:
            fh.write("\n".join(img_test))
        # Write validation file
        with open(os.path.join(self.out_path, "valid.txt"), "w") as fh:
            fh.write("\n".join(img_valid))
        pass

    def __create_obj_dir(self, base_dir: str):
        """
        Crea un directorio obj dentro del directorio de salida si este no existe ya.
        :param base_dir: Directorio base donde crear el directorio obj.
        :return: El path al directorio obj de salida.
        """
        obj_dir = os.path.join(base_dir, self.obj_dir_name)
        if not os.path.exists(obj_dir):
            os.makedirs(obj_dir)
        return obj_dir

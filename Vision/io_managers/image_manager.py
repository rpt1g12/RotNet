import os

import numpy as np
from PIL import Image
import cv2

from Vision import Sample
from Vision.io_managers import Manager
from Vision.sample import Sample_List, Sample_Generator
from Vision.utils import image_utils
from Vision.utils.parallelisation import threadsafe_generator


class ImageManager(Manager):
    """Clase basica para cargar imagenes sin etiquetar desde un directorio."""

    def __init__(self, in_path: str, batch_size=32, shuffle=False, seed=0):
        self.batch_size = batch_size
        self.in_path = in_path
        self.file_list = list(filter(image_utils.is_image, os.listdir(self.in_path)))
        super(ImageManager, self).__init__(in_path, None, self.file_list,
                                           batch_size, shuffle, seed)

    def clone(self) -> 'ImageManager':
        return ImageManager(
            in_path=self.in_path,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            seed=self.seed
        )

    def set_in_path(self, in_path: str):
        self.in_path = in_path
        self.set_file_list(list(filter(image_utils.is_image, os.listdir(self.in_path))))

    def read_sample(self, n) -> Sample:
        if n < 0:
            n = np.random.randint(0, len(self.file_list))
        img_name = self.file_list[n]
        img_path = os.path.join(self.in_path, img_name)
        return self.__to_sample(n, img_path)

    def write_sample(self, sample: Sample, **kwargs) -> int:
        print("This manager does not write samples.")
        return 0

    @threadsafe_generator
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        if batch_size > 0:
            self.set_batch_size(batch_size)
        assert self.in_path is not None, "No se ha especificado un path de donde leer las Imagenes"
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

    def write_samples(self, samples: Sample_List, **kwargs) -> int:
        self.write_sample(samples[0], **kwargs)
        return 0

    def __file_generator(self):
        return image_utils.image_path_generator(self.in_path)

    def __to_sample(self, sample_id, file):
        """
        Carga una imagen y la convierte en un objeto Muestra sin anotaciones.
        :param sample_id: Id de la muestra.
        :param file: Nombre del fichero imagen.
        :return: Muestra sin anotaciones de la imagen.
        """
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        return Sample(sample_id, os.path.basename(file), Image.fromarray(img))

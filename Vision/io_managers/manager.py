import abc

from keras.preprocessing.image import Iterator
from matplotlib import pyplot as plt

from Vision.sample import Sample, Sample_Generator, Sample_List


class Manager(Iterator, metaclass=abc.ABCMeta):
    """
    Interfaz para managers I/O
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'write_samples') and
                callable(subclass.write_samples) and
                hasattr(subclass, 'sample_generator') and
                callable(subclass.sample_generator) and
                hasattr(subclass, 'read_sample') and
                callable(subclass.read_sample) and
                hasattr(subclass, 'write_sample') and
                callable(subclass.write_sample) or
                NotImplemented
        )

    def __init__(self,
                 in_path, out_path, file_list,
                 batch_size, shuffle, seed
                 ):

        self.in_path = in_path
        self.out_path = out_path
        self.file_list = file_list
        super().__init__(len(file_list), batch_size, shuffle, seed)

    @abc.abstractmethod
    def clone(self) -> "Manager":
        """
        Creates a new copy of the Manager class
        :return: new Manager class
        """
        raise NotImplemented

    def get_out_path(self) -> str:
        return self.out_path

    def set_out_path(self, out_path: str) -> None:
        """
        Sets output path for the man_in.
        :param out_path: Output path
        """
        self.out_path = out_path

    def get_in_path(self) -> str:
        return self.in_path

    def set_in_path(self, in_path: str) -> None:
        """
        Sets input path for the man_in.
        :param in_path: Input path
        """
        self.in_path = in_path

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets the batch size
        :param batch_size: Batch size.
        """
        self.batch_size = batch_size

    def set_file_list(self, file_list: list) -> None:
        """
        Sets the list of files available to manager
        :param file_list: list of files
        :return:
        """
        self.n = len(file_list)
        self.file_list = file_list

    def get_file_list(self) -> list:
        return self.file_list

    def random_check(self, zoom=False, checks=10, *args, **kwargs):
        for _ in range(checks):
            plt.close()
            sample = self.read_sample(-1)
            print(sample.name)
            if zoom:
                sample = sample.zoom_to_annotations()
            sample.plot(*args, **kwargs)
            plt.waitforbuttonpress()

    @abc.abstractmethod
    def read_sample(self, n: int) -> Sample:
        """
        Extrae la enesima muestra con imagen y anotaciones.
        :param n: Id de la muestra.
        :return: Muestra con imagen y anotaciones
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_sample(self, sample: Sample, write_image=False) -> int:
        """
        Guarda la enesima muestra con imagen y anotaciones.
        :param write_image: True si se quiere guardar la imagen tambien
        :param sample: Muestra a guardar.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def sample_generator(self, batch_size: int = 0) -> Sample_Generator:
        """
        Extrae todas las muestra con imagen y anotaciones.
        :param batch_size: Tamano del batch.
        :return: Lista de muestras con imagen y anotaciones
        """
        raise NotImplementedError

    @abc.abstractmethod
    def write_samples(self, samples: Sample_List, write_image=False) -> int:
        """
        Guarda la una lista de muestras con imagen y anotaciones.
        :param samples: Muestra a guardar.
        :param write_image: True si queremos guardar la imagen, False en caso contrario
        """
        raise NotImplementedError

    def _get_batches_of_transformed_samples(self, index_array) -> Sample_List:
        batch_size = len(index_array)
        samples = list()
        for i in range(batch_size):
            samples.append(self.read_sample(i))
        return samples


import abc
import json
import os

from sklearn.model_selection import train_test_split

from Vision.io_managers.manager import Manager
from Vision.sample import Sample


class DatasetSplitter(metaclass=abc.ABCMeta):

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'get_target') and
                callable(subclass.get_target) or
                NotImplemented
        )

    def __init__(self,
                 manager: Manager,
                 val_split: float = None,
                 test_split: float = 0.1
                 ):
        """
        Crea splits de train y test (validacion opcional) devolviendo managers separados
        para cada split.
        :param manager: Manager original que se divide en varios splits.
        :param val_split: Porcentage del split de validacion. Por defecto no se utiliza, y
        la muestra se divide en train y test.
        :param test_split: Porcentage del split de test.
        """

        self.split_dict = None
        self.manager = manager
        self.val_split = val_split
        self.test_split = test_split

    @abc.abstractmethod
    def get_target(self, sample: Sample) -> int:
        """
        This method gets the target specific to the dataset
        :return: target class
        """
        raise NotImplemented

    def read_split_file(self, split_file_path: str):
        """
        This method reads the defined split_file
        :return: dict with splits
        """
        with open(split_file_path, 'r') as file:
            split_dict = json.load(file)
        self.split_dict = split_dict
        self._read_splits_from_dict()

    def get_train(self) -> Manager:
        return self._get_new_manager('train')

    def get_test(self) -> Manager:
        return self._get_new_manager('test')

    def get_val(self) -> Manager:
        return self._get_new_manager('val')

    def get_split_file_list(self, dataset='train'):
        assert dataset in ['train', 'test', 'val'], "dataset tiene que ser 'train', 'val' o 'test'"
        return self.split_dict[dataset]['file_list']

    def get_split_target_list(self, dataset='train'):
        assert dataset in ['train', 'test', 'val'], "dataset tiene que ser 'train', 'val' o 'test'"
        return self.split_dict[dataset]["target_list"]

    def split_dataset(self, split_file_path: str = None) -> None:
        """
        Metodo para partir un data set en train/val/test
        :val_split: proporción de casos de validación
        :val_split: proporción de casos de test
        :return: diccionario de splits
        """
        self.split_dict = self._splitter()
        if split_file_path is None:
            split_file_path = os.path.join(self.manager.in_path, "dataset_split.json")
        self._save_split_file(split_file_path)

    def _splitter(self) -> dict:
        """
        This method defines the split of a dataset
        :return: dictionary with list of filenames for each split (train, test, val)
        """
        assert self.test_split is not None, "test_split no puede ser None"
        x = list()
        y = list()
        for i in range(len(self.manager.file_list)):
            sample = self.manager.read_sample(i)
            x.append(sample.name)
            y.append(self.get_target(sample))

        if self.val_split is None:
            test_split = self.test_split
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.test_split, stratify=y)
            self.val_split = 0
            x_val = []
            y_val = []
        else:
            test_split = self.val_split + self.test_split
            test_val_split = self.test_split / test_split
            x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size=test_split, stratify=y)
            x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=test_val_split,
                                                            stratify=y_test_val)

        split_dict = dict(train=dict(file_list=x_train, target_list=y_train),
                          val=dict(file_list=x_val, target_list=y_val),
                          test=dict(file_list=x_test, target_list=y_test),
                          splits=dict(val=self.val_split, test=self.test_split, train=1 - test_split))

        return split_dict

    def _read_splits_from_dict(self):
        self.val_split = self.split_dict['splits']['val']
        self.test_split = self.split_dict['splits']['test']

    def _save_split_file(self, split_file_path: str):
        """
        This method saves the split dict to the defined split_file
        :return: dict with splits
        """
        with open(split_file_path, 'w') as file:
            json.dump(self.split_dict, file)

    def _get_new_manager(self, dataset: str) -> Manager:
        new_manager = self.manager.clone()
        new_manager.set_file_list(self.get_split_file_list(dataset))
        return new_manager

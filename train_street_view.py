from __future__ import print_function

import os
import sys
from concurrent.futures import Future
import concurrent.futures
from tqdm import tqdm
from typing import Callable, Iterable, List

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import ResNet50
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Dense, Flatten
from keras.optimizers import SGD

from utils import REGEX_IMG

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import angle_error, RotNetDataGenerator
import argparse

SPLIT_NAMES = ["train", "val", "test"]


def parallelize_with_thread_pool(f: Callable, iterable: Iterable, size: int = None) -> List[Future]:
    """
    Parallelises de execution of the incomming function over the iterable using threadding.
    :param f: Function to execute in parallel.
    :param iterable: Iterable to map the function to.
    :param size: Size of the iterable (Optional). Defaults to None
    :return: List of Futures
    """
    with tqdm(total=size) as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(f, batch) for batch in iterable]
            for _ in concurrent.futures.as_completed(futures):
                pbar.update(1)
    return futures


def get_move_files_function(src, split_name):
    # Destination folder
    dst = os.path.join(src, split_name)
    # If it does not create, create it.
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Helper function to move the files from src to dst
    def move_file(file_name: str):
        os.rename(os.path.join(src, file_name), os.path.join(dst, file_name))

    # Function that will move the image file as well as its voc
    def f(image_path: str):
        image_file = os.path.basename(image_path)
        voc_file = REGEX_IMG.sub(r"\1xml", image_file)
        move_file(image_file)
        move_file(voc_file)

    return f


def get_filenames(src):
    return [os.path.join(r, f) for r, d, files in os.walk(src) for f in files if REGEX_IMG.match(f)]


class SmartFormatter(argparse.HelpFormatter):
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RotNet-Trainer",
        formatter_class=SmartFormatter
    )
    parser.add_argument("--source", "-s",
                        type=str,
                        help="""R|
Path al directorio de donde importar las imagenes. Este directorio debe
a su vez contener dos carpetas `train` y `val` que contienen las imagenes
para entrenar y validar el modelo.
                        """,
                        required=True
                        )
    parser.add_argument("--name", "-n",
                        type=str,
                        help="Nombre que queremos que tenga el modelo.",
                        default="RotNet"
                        )
    parser.add_argument("--batch_size", "-b",
                        type=int,
                        help="Tamaño de los batches.",
                        default=32
                        )
    parser.add_argument("--epochs", "-e",
                        type=int,
                        help="Numero de épocas a entrenar.",
                        default=50
                        )
    parser.add_argument("--workers", "-w",
                        type=int,
                        help="Numero de workers a utilizar para los generadores.",
                        default=8
                        )
    return parser.parse_args()


class RotNetTrainer(object):
    """Docstring"""

    def __init__(self,
                 sample_source_path,
                 model_name,
                 splits=[0.7, 0.15, 0.15]
                 ):
        super(RotNetTrainer, self).__init__()
        self.sample_source_path = sample_source_path
        self.create_train_split(sample_source_path, splits)

        self.train_filenames = get_filenames(os.path.join(sample_source_path, "train"))
        self.test_filenames = get_filenames(os.path.join(sample_source_path, "test"))
        print(len(self.train_filenames), 'train samples')
        print(len(self.test_filenames), 'test samples')

        self.model_name = model_name

        # number of classes
        nb_classes = 360
        # input image shape
        input_shape = (224, 224, 3)
        self.model = self._build(input_shape, nb_classes)

        # Create output folder
        self.output_folder = 'saved_models'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def create_train_split(self, voc_in_path: str, splits=[0.7, 0.15, 0.15]):
        assert sum(splits) == 1.0, "the sum of the splits should be 1.0"
        for s in splits:
            assert s > 0, "the splits % must be positive"
        src = voc_in_path
        # Get filenames of images
        files = [os.path.join(r, f) for r, d, files in os.walk(src) for f in files if REGEX_IMG.match(f)]

        idx0 = 0
        n_samples = len(files)
        for i in range(len(splits)):
            split_name = SPLIT_NAMES[i]
            self.sample_sources[split_name] = os.path.join(src, split_name)
            move_function = get_move_files_function(src, split_name)
            # Select indices to move
            idx1 = int(n_samples * splits[i]) + idx0 if (i < (len(splits) - 1)) else n_samples
            parallelize_with_thread_pool(move_function, files[idx0:idx1], idx1 - idx0)
            # Update left bound of indices
            idx0 = idx1

    def _build(self, input_shape, nb_classes):
        # load base model
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape)

        # append classification layer
        x = base_model.output
        x = Flatten()(x)
        final_output = Dense(nb_classes, activation='softmax', name='fc360')(x)

        # create the new model
        model = Model(inputs=base_model.input, outputs=final_output)
        model.summary()
        # model compilation
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.01, momentum=0.9),
                      metrics=[angle_error])
        return model

    def get_callbacks(self):
        # callbacks
        monitor = "val_angle_error"
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(self.output_folder, self.model_name + '.hdf5'),
            monitor=monitor,
            save_best_only=True
        )
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        early_stopping = EarlyStopping(monitor=monitor, patience=5)
        tensorboard = TensorBoard()

        return [checkpointer, reduce_lr, early_stopping, tensorboard]

    def train(self, batch_size, nb_epochs, workers):
        # training loop
        self.model.fit_generator(
            RotNetDataGenerator(
                self.train_filenames,
                input_shape=self.input_shape,
                batch_size=batch_size,
                preprocess_func=preprocess_input,
                crop_center=True,
                crop_largest_rect=True,
                shuffle=True
            ),
            steps_per_epoch=len(self.train_filenames) / batch_size,
            epochs=nb_epochs,
            validation_data=RotNetDataGenerator(
                self.test_filenames,
                input_shape=self.input_shape,
                batch_size=batch_size,
                preprocess_func=preprocess_input,
                crop_center=True,
                crop_largest_rect=True
            ),
            validation_steps=len(self.test_filenames) / batch_size,
            callbacks=self.get_callbacks(),
            workers=workers
        )


if __name__ == '__main__':
    args = parse_args()

    rotnet = RotNetTrainer(args.source, args.name)
    print(args.source, args.name, args.batch_size, args.epochs, args.workers)

import os
import re
from typing import Dict, Tuple

import numpy as np
from Vision.io_managers import VOC, SampleAugmenter
from Vision.models.classification_model import ClassificationModel
from Vision.utils.parallelisation import parallelize_with_thread_pool
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.optimizers import SGD

from utils import angle_error, REGEX_IMG
from keras.applications import ResNet50
from keras.layers import Flatten, Dense
from keras import Model as _kModel

from vision_generator import VocRotGenerator

OUTPUT_FOLDER = 'saved_models'
SPLIT_NAMES = ["train", "val", "test"]


def create_subfolders(src, n_splits):
    for dir_name in SPLIT_NAMES[:n_splits]:
        dir_path = os.path.join(src, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


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


class RotNet(ClassificationModel):
    """Model that predicts rotation angle of an image"""

    def predict_image_array(self, img_array: np.ndarray) -> Dict[str, float]:
        pass

    def __init__(self, model_name: str, deg_resolution: int, make_grayscale: bool,
                 input_shape: Tuple[int, int], output_folder=OUTPUT_FOLDER):
        super(RotNet, self).__init__()
        # Make sure the output path is available
        if not os.path.exists(OUTPUT_FOLDER):
            os.makedirs(OUTPUT_FOLDER)
        self.model_name = model_name
        self.output_folder = output_folder
        self.deg_resolution = deg_resolution
        assert 360 % deg_resolution == 0, "Resolution must be a factor of 360"
        self.n_classes = int(360 / self.deg_resolution)
        self.make_grayscale = make_grayscale
        self.color_channels = 1 if make_grayscale else 3
        self.input_shape = input_shape
        self.kmodel = self.build()
        self.sample_sources = dict()

    def build(self) -> _kModel:
        input_shape = self.input_shape + (self.color_channels,)
        # load base model
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape)

        # append classification layer
        x = base_model.output
        x = Flatten()(x)
        final_output = Dense(self.n_classes, activation='softmax', name='rotnet-output')(x)

        # create the new model
        model = _kModel(inputs=base_model.input, outputs=final_output)
        model.summary()

        return model

    def compile(self):
        # model compilation
        self.kmodel.compile(loss='categorical_crossentropy',
                            optimizer=SGD(lr=0.01, momentum=0.9),
                            metrics=[angle_error])

    def get_callbacks(self):
        # callbacks
        monitor = "val_angle_error"
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(OUTPUT_FOLDER, self.model_name + '.hdf5'),
            monitor=monitor,
            save_best_only=True
        )
        reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=3)
        early_stopping = EarlyStopping(monitor=monitor, patience=5)
        tensorboard = TensorBoard()

        return [checkpointer, reduce_lr, early_stopping, tensorboard]

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

    def train(self, epochs, batch_size, n_aug=0,
              train_source: str = None, val_source: str = None,
              preprocessing_function=None,
              workers: int = 2, multiprocessing: bool = False):
        train_path = train_source or self.sample_sources.get(SPLIT_NAMES[0])
        val_path = val_source or self.sample_sources.get(SPLIT_NAMES[1])

        assert train_path is not None, "You must set the train_source. Either call `create_train_split` or provide" \
                                      "a path to the train folder"
        assert val_path is not None, "You must set the val_source. Either call `create_train_split` or provide" \
                                     "a path to the val folder"
        # Define generators
        train_generator = VocRotGenerator(
            manager=VOC(in_path=train_path, batch_size=batch_size),
            input_shape=self.input_shape,
            deg_resolution=self.deg_resolution,
            batch_size=batch_size,
            preprocessing_function=preprocessing_function,
            make_grayscale=self.make_grayscale,
            n_aug=n_aug
        )

        val_generator = VocRotGenerator(
            manager=VOC(in_path=val_path, batch_size=batch_size),
            input_shape=self.input_shape,
            deg_resolution=self.deg_resolution,
            batch_size=batch_size,
            preprocessing_function=preprocessing_function,
            make_grayscale=self.make_grayscale,
            n_aug=n_aug
        )

        # Compile model
        self.compile()

        # training loop
        self.kmodel.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=self.get_callbacks(),
            workers=workers,
            use_multiprocessing=multiprocessing
        )

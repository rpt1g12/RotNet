import os
from typing import Dict, Tuple

import cv2
import keras.backend as K
import numpy as np
from Vision import Sample
from Vision.io_managers import VOC
from Vision.models.classification_model import ClassificationModel
from Vision.utils.parallelisation import parallelize_with_thread_pool
from keras import Model as _kModel, Input
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.applications.imagenet_utils import preprocess_input as imagenet_pre
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import load_model
from keras.optimizers import Optimizer

from utils import REGEX_IMG, angle_difference
from vision_generator import VocRotGenerator

OUTPUT_FOLDER = 'saved_models'
SPLIT_NAMES = ["train", "val", "test"]

BACKBONES = ["resnet50", "simple"]


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
    def move_file(file_name_tuple: Tuple[str, str]):
        src_tuple = (src,) + file_name_tuple
        src_path = os.path.join(*src_tuple)
        dst_path = os.path.join(dst, file_name_tuple[1])
        if src_path != dst_path:
            os.rename(src_path, dst_path)

    # Function that will move the image file as well as its voc
    def f(image_tuple_path: Tuple[str, str]):
        image_file = image_tuple_path[1]
        voc_file = REGEX_IMG.sub(r"\1xml", image_file)
        move_file((image_tuple_path[0], image_file))
        move_file((image_tuple_path[0], voc_file))

    return f


class RotNet(ClassificationModel):
    """Model that predicts rotation angle of an image"""

    def __init__(self, model_name: str,
                 deg_resolution: int,
                 make_grayscale: bool,
                 input_shape: Tuple[int, int],
                 backbone="simple",
                 output_folder=OUTPUT_FOLDER):
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
        assert backbone in BACKBONES, "backbone should be one of: \n\t{}".format(BACKBONES)
        self.backbone = backbone
        self.kmodel = self.build()
        self.sample_sources = dict()

    def get_angle_error_function(self):
        factor = self.deg_resolution

        def angle_error(y_true, y_pred):
            """
            Calculate the mean diference between the true angles
            and the predicted angles. Each angle is represented
            as a binary vector.
            """
            diff = angle_difference(K.argmax(y_true) * factor, K.argmax(y_pred) * factor)
            return K.mean(K.cast(K.abs(diff), K.floatx()))

        return angle_error

    def get_preprocessing_function(self):
        if self.backbone == "resnet50":
            return resnet_pre
        else:
            return imagenet_pre

    def build(self):
        if self.backbone == "resnet50":
            return self.restnet_build()
        else:
            return self.simple_build()

    def simple_build(self):
        # size of pooling area for max pooling
        pool_size = (2, 2)
        # convolution kernel size
        kernel_size = (3, 3)

        # number of classes
        input_shape = self.input_shape + (self.color_channels,)
        input_layer = Input(shape=input_shape)
        x = Conv2D(64, kernel_size, activation='relu', padding="same")(input_layer)
        x = MaxPooling2D(pool_size=pool_size, padding="same")(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=pool_size, padding="same")(x)
        x = Dropout(0.25)(x)
        x = Conv2D(16, kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=pool_size, padding="same")(x)
        x = Dropout(0.25)(x)
        x = Conv2D(8, kernel_size, activation='relu')(x)
        x = MaxPooling2D(pool_size=pool_size, padding="same")(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.25)(x)
        output_layer = Dense(self.n_classes, activation='softmax', name="rotnet-output")(x)

        model = _kModel(inputs=input_layer, outputs=output_layer)

        model.summary()
        return model

    def restnet_build(self) -> _kModel:
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

    def compile(self, optimizer: Optimizer = None):
        angle_error = self.get_angle_error_function()
        if optimizer is not None:
            self.kmodel.compile(loss='categorical_crossentropy',
                                optimizer=optimizer,
                                metrics=[angle_error])
        else:
            self.kmodel.compile(loss='categorical_crossentropy',
                                optimizer='adam',
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
        # Get filenames of images recursively and relative folder with respect to src
        files = [(os.path.relpath(r, src), f) for r, d, files in os.walk(src) for f in files if REGEX_IMG.match(f)]
        # Sort files by their basename
        files.sort(key=lambda t: t[1])

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

    def load(self, hdf5_path):
        self.kmodel = load_model(hdf5_path, compile=False)
        self.compile()

    def train(self, epochs, batch_size, optimizer: Optimizer, n_aug=0,
              train_source: str = None, val_source: str = None,
              **kwargs
              ):
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
            preprocessing_function=self.get_preprocessing_function(),
            make_grayscale=self.make_grayscale,
            n_aug=n_aug
        )

        val_generator = VocRotGenerator(
            manager=VOC(in_path=val_path, batch_size=batch_size),
            input_shape=self.input_shape,
            deg_resolution=self.deg_resolution,
            batch_size=batch_size,
            preprocessing_function=self.get_preprocessing_function(),
            make_grayscale=self.make_grayscale,
            n_aug=0,
            shuffle=False
        )

        # Compile model
        self.compile(optimizer=optimizer)

        # training loop
        self.kmodel.fit_generator(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=len(val_generator),
            callbacks=self.get_callbacks(),
            **kwargs
        )

    def preprocess_sample(self, sample: Sample) -> np.ndarray:
        return self.preprocess_image_array(sample.get_img_arr())

    def preprocess_image_array(self, img_array: np.ndarray) -> np.ndarray:
        # Ressize image to fit expected size for VGG and pre-process it
        resized_img = cv2.resize(img_array, self.input_shape)
        vector_img = np.expand_dims(resized_img, axis=0)
        pre = self.get_preprocessing_function()
        return pre(vector_img)

    def predict_image_array(self, img_array: np.ndarray) -> Dict[str, float]:
        scores = self.kmodel.predict(self.preprocess_image_array(img_array))
        theta = np.argmax(scores) * self.deg_resolution
        return {"theta": theta}

    def predict_sample_append(self, sample: Sample) -> Sample:
        """
        Devuelve la muestra de entrada con etiquetas en el campo metadata
        :param sample: Muestra de entrada
        :return: Muestra con nuevas etiquetas predichas.
        """
        theta = self.predict_sample(sample).get("theta") or 0
        prediction = sample.clone()
        return prediction.set_rotation(theta).apply_rotation()

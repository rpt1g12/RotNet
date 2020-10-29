import os
from typing import Dict, Tuple

import cv2
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras import Model as _kModel, Input
from keras.applications import ResNet50
from keras.applications.imagenet_utils import preprocess_input as imagenet_pre
from keras.applications.resnet50 import preprocess_input as resnet_pre
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, BatchNormalization, Activation, \
    GlobalAvgPool2D, Add
from keras.models import load_model
from keras.optimizers import Optimizer

from Vision import Sample
from Vision.io_managers import Manager
from Vision.models.classification_model import ClassificationModel
from Vision.utils.parallelisation import parallelize_with_thread_pool
from utils import REGEX_IMG
from vision_generator import RotNetManager

OUTPUT_FOLDER = 'saved_models'
SPLIT_NAMES = ["train", "val", "test"]

BACKBONES = ["resnet50", "custom"]


def initial_block(filters, kernel_size):
    def f(input):
        x = Conv2D(
            filters,
            kernel_size,
            strides=2,
            padding="same",
            use_bias=False
        )(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)
        return x

    return f


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


def softargmax(x, nb_classes, beta=1e10):
    x_range = tf.range(nb_classes, dtype=x.dtype)
    return tf.reduce_sum(tf.nn.softmax(x * beta) * x_range, axis=-1)


def residual_block(filters, stride, n):
    def f(x):
        z = Conv2D(
            filters,
            3,
            strides=stride,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            name="conv_resblk_{}_main_1".format(n)
        )(x)
        z = BatchNormalization(name="bn_resblk_{}_main_1".format(n))(z)
        z = Activation("relu", name="act_resblk_{}_main_1".format(n))(z)
        z = Conv2D(
            filters,
            3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name="conv_resblk_{}_main_2".format(n)
        )(z)
        z = BatchNormalization(name="bn_resblk_{}_main_2".format(n).format(n))(z)
        z_skip = x
        if stride > 1:
            z_skip = Conv2D(
                filters,
                1,
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_initializer="he_normal",
                name="conv_resblk_{}_skip".format(n)
            )(z_skip)
            z_skip = BatchNormalization(name="bn_resblk_{}_skip".format(n))(z_skip)
        z_add = Add(name="add_resblk_{}".format(n))([z, z_skip])
        out = Activation("relu", name="act_resblk_{}".format(n))(z_add)
        return out

    return f


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
                 backbone: str,
                 regression: False,
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
        self.regression = regression
        self.kmodel = None
        self.train_generator = None
        self.val_generator = None

    def get_angle_error_function(self):
        factor = self.deg_resolution

        def angle_error_classification(y_true, y_pred):
            """
            Calculate the mean diference between the true angles
            and the predicted angles. Each angle is represented
            as a binary vector.
            """
            diff = angle_difference(K.argmax(y_true) * factor, K.argmax(y_pred) * factor)
            return K.mean(K.cast(K.abs(diff), K.floatx()))

        def classification_loss(y_true, y_pred):
            diff = angle_difference(softargmax(y_true, self.n_classes) * factor, softargmax(y_pred, self.n_classes) * factor)
            return K.mean(K.cast(K.abs(diff), K.floatx()))

        def regression_loss(y_true, y_pred):
            """
            Calculate the mean diference between the true angles
            and the predicted angles.
            """
            return K.mean(angle_difference(y_true * 360, y_pred * 360) / 360)

        def angle_error_regression(y_true, y_pred):
            return regression_loss(y_true, y_pred) * 360

        if self.regression:
            return angle_error_regression, regression_loss
        else:
            return angle_error_classification, classification_loss

    def get_preprocessing_function(self):
        if self.backbone == "resnet50":
            return resnet_pre
        else:
            return imagenet_pre

    def build(self):
        if self.backbone == "resnet50":
            self.kmodel = self.restnet_build()
        else:
            self.kmodel = self.custom_build()

    def custom_build(self):
        # Input layer
        input_shape = self.input_shape + (self.color_channels,)

        input_layer = Input(shape=input_shape)

        # Initial filter size
        filt_0 = 16
        x = initial_block(filt_0,7)(input_layer)

        # ResidualBlocks
        res_block_depths = [2, 2, 2, 2]
        res_filters = [[filt_0 * 2 ** i] * depth for i, depth in enumerate(res_block_depths)]
        for i in range(len(res_block_depths)):
            ii = i + 1
            for j, filters in enumerate(res_filters[i]):
                jj = j + 1
                strides = int(filters / filt_0)
                x = residual_block(filters, strides, ii * 10 + jj)(x)
                filt_0 = filters

        x = GlobalAvgPool2D()(x)
        # x = Dense(128, activation="relu")(x)
        # x = Dropout(0.2)(x)
        # x = Dense(256, activation="relu")(x)
        # x = Dropout(0.2)(x)
        if self.regression:
            output_layer = Dense(1, activation="sigmoid", name="rotnet-output")(x)
        else:
            output_layer = Dense(self.n_classes, activation='softmax', name="rotnet-output")(x)

        model = _kModel(inputs=input_layer, outputs=output_layer)
        return model

    def restnet_build(self) -> _kModel:
        input_shape = self.input_shape + (self.color_channels,)
        # load base model
        base_model = ResNet50(weights='imagenet', include_top=False,
                              input_shape=input_shape)

        # append classification layer
        x = base_model.output
        x = Flatten()(x)
        if self.regression:
            output_layer = Dense(1, activation="sigmoid", name="rotnet-output")(x)
        else:
            output_layer = Dense(self.n_classes, activation='softmax', name="rotnet-output")(x)

        # create the new model
        model = _kModel(inputs=base_model.input, outputs=output_layer)
        return model

    def compile(self, optimizer: Optimizer = None):
        angle_error, loss = self.get_angle_error_function()
        if optimizer is not None:
            self.kmodel.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=[angle_error])
        else:
            self.kmodel.compile(loss=loss,
                                optimizer='adam',
                                metrics=[angle_error])

    def get_callbacks(self):
        # callbacks
        if self.regression:
            monitor = "val_angle_error_regression"
        else:
            monitor = "val_angle_error_classiffication"
        file_name = "{}-{}".format(self.model_name, self.backbone)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(OUTPUT_FOLDER, "{}.hdf5".format(file_name)),
            monitor=monitor,
            save_best_only=True
        )
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", patience=3)
        early_stopping = EarlyStopping(monitor=monitor, patience=10)
        tensorboard = TensorBoard(log_dir=os.path.join("logs", file_name))

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
        np.random.shuffle(files)

        idx0 = 0
        n_samples = len(files)
        for i in range(len(splits)):
            split_name = SPLIT_NAMES[i]
            move_function = get_move_files_function(src, split_name)
            # Select indices to move
            idx1 = int(n_samples * splits[i]) + idx0 if (i < (len(splits) - 1)) else n_samples
            parallelize_with_thread_pool(move_function, files[idx0:idx1], idx1 - idx0)
            # Update left bound of indices
            idx0 = idx1

    def load(self, hdf5_path, compile_model=True):
        self.kmodel = load_model(hdf5_path, compile=False)
        if compile_model:
            self.compile()

    def train(self, epochs, batch_size,
              train_man: Manager, val_man: Manager,
              n_aug=0, fixed=False, **kwargs
              ):

        train_man.set_batch_size(batch_size)
        val_man.set_batch_size(batch_size)

        # Define generators
        self.train_generator = RotNetManager(
            manager=train_man,
            input_shape=self.input_shape,
            deg_resolution=self.deg_resolution,
            batch_size=batch_size,
            preprocessing_function=self.get_preprocessing_function(),
            make_grayscale=self.make_grayscale,
            fixed=fixed,
            n_aug=n_aug,
            regression=self.regression,
            shuffle=True
        )

        self.val_generator = RotNetManager(
            manager=val_man,
            input_shape=self.input_shape,
            deg_resolution=self.deg_resolution,
            batch_size=batch_size,
            preprocessing_function=self.get_preprocessing_function(),
            make_grayscale=self.make_grayscale,
            fixed=True,
            n_aug=0,
            regression=self.regression,
            shuffle=False
        )

        # training loop
        self.kmodel.fit_generator(
            self.train_generator,
            steps_per_epoch=len(self.train_generator),
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=len(self.val_generator),
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
        return prediction.set_rotation(-theta)

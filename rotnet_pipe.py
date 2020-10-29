import os
from typing import Tuple

from Vision.io_managers import Manager
from Vision.pipelines.classification_pipe import ClassificationPipe

from rotnet import RotNet


class RotnetPipe(ClassificationPipe):
    """Pipeline para ejecutar el modelo RotNet"""

    def __init__(self, hdf5: str, deg_resolution: int, backbone: str, sample_manager: Manager,
                 make_grayscale: bool = False,
                 input_shape: Tuple[int, int] = (224, 224)):
        name = os.path.basename(hdf5).replace(".hdf5", "")
        rotnet = RotNet(
            model_name=name,
            deg_resolution=deg_resolution,
            make_grayscale=make_grayscale,
            input_shape=input_shape,
            regression=False,
            backbone=backbone
        )
        rotnet.load(hdf5)
        super(RotnetPipe, self).__init__(sample_manager, rotnet)

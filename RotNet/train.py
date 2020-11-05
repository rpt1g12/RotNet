import os
from datetime import datetime as dt

from Vision.io_managers import ImageManager
from keras.optimizers import SGD

from RotNet.rotnet import RotNet

if __name__ == "__main__":
    today = dt.date(dt.now()).__str__()
    model_name = "RotNet-0-rotation2-{}".format(today)
    rotnet = RotNet(
        model_name=model_name,
        deg_resolution=45,
        make_grayscale=False,
        input_shape=(224, 224),
        backbone="mobilenet",
        regression=False
    )

    input_path = "/home/henrypaul/LDARPT/datasets/rotation_2"
    # rotnet.create_train_split(input_path, [0.9, 0.1])
    rotnet.build()
    # rotnet.load("saved_models/RotNet-8-rotation2-2020-10-29-custom.hdf5", compile_model=False)
    rotnet.compile(SGD(lr=1e-3, momentum=0.9, decay=0.001))

    rotnet.train(
        train_man=ImageManager(
            in_path=os.path.join(input_path, "train")
        ),
        val_man=ImageManager(
            in_path=os.path.join(input_path, "val"))
        ,
        epochs=50,
        batch_size=64,
        n_aug=0,
        fixed=False,
        workers=32,
        max_queue_size=64,
        initial_epoch=0
    )

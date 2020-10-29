import os
from datetime import datetime as dt

from Vision.io_managers import ImageManager
from keras.optimizers import SGD, Adam

from rotnet import RotNet

if __name__ == "__main__":
    today = dt.date(dt.now()).__str__()
    model_name = "RotNet-6.2-{}".format(today)
    rotnet = RotNet(
        model_name=model_name,
        deg_resolution=2,
        make_grayscale=False,
        input_shape=(256, 256),
        backbone="custom"
    )

    input_path = "/home/henrypaul/LDARPT/datasets/rotation_2"
    rotnet.create_train_split(input_path, [0.8, 0.2])
    # rotnet.build()
    rotnet.load("saved_models/RotNet-6.1-2020-10-29-custom.hdf5", compile_model=False)
    rotnet.compile(Adam())

    rotnet.train(
        train_man=ImageManager(
            in_path=os.path.join(input_path, "train")
        ),
        val_man=ImageManager(
            in_path=os.path.join(input_path, "val"))
        ,
        epochs=50,
        batch_size=32,
        n_aug=0,
        fixed=False,
        workers=2,
        max_queue_size=64,
        use_multiprocessing=True
    )

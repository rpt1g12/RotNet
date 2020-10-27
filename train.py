import os
from datetime import datetime as dt

from Vision.io_managers import ImageManager
from keras.optimizers import SGD, Adam

from rotnet import RotNet

if __name__ == "__main__":
    today = dt.date(dt.now()).__str__()
    model_name = "RotNet-{}".format(today)
    rotnet = RotNet(
        model_name=model_name,
        deg_resolution=2,
        make_grayscale=False,
        input_shape=(256, 256),
        backbone="custom"
    )

    input_path = "/home/henrypaul/LDARPT/datasets/rotation"
    # rotnet.create_train_split(input_path, [0.8, 0.1, 0.1])

    rotnet.train(
        train_man=ImageManager(
            in_path=os.path.join(input_path, "train")
        ),
        val_man=ImageManager(
            in_path=os.path.join(input_path, "val"))
        ,
        epochs=50,
        batch_size=64,
        optimizer=Adam(),
        n_aug=0,
        workers=32,
        max_queue_size=128
    )

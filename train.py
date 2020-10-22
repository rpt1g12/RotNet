from datetime import datetime as dt

from keras.optimizers import SGD

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

    input_path = "/home/henrypaul/LDARPT/rotnet/autoscout"
    rotnet.create_train_split(input_path, [0.8, 0.1, 0.1])

    rotnet.train(
        epochs=50,
        batch_size=64,
        optimizer=SGD(lr=0.01, momentum=0.9, decay=0.001),
        n_aug=0,
        workers=32,
        max_queue_size=128
    )

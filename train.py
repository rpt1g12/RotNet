import os

from keras.optimizers import SGD

from rotnet import RotNet

if __name__ == "__main__":
    rotnet = RotNet(
        model_name="RotNet-Simple-2020-10-21",
        deg_resolution=5,
        make_grayscale=False,
        input_shape=(224, 224),
        use_resnet=False
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

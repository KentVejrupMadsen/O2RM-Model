#/usr/bin/python
from keras.layers   \
    import          \
    Conv2D,         \
    MaxPooling2D,   \
    Rescaling


def input_layer(
    width: int,
    height: int,
    channels: int,
    scale_by: float = 1.0/127.5,
    offset: float = -1.0
) -> list:
    r: list = list()

    r.append(
        Rescaling(
            scale_by,
            input_shape=(
                height,
                width,
                channels
            ),
            offset=offset
        )
    )

    for i in range(4):
        r.append(
            Conv2D(
                2,
                channels,
                padding='same',
                activation='relu',
            )
        )

    r.append(
        MaxPooling2D(
            (2, 2)
        )
    )

    return r


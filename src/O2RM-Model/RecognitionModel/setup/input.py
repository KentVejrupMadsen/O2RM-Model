#/usr/bin/python
from keras.layers   \
    import          \
    Conv2D,         \
    MaxPooling2D,   \
    Rescaling


def input_layer(
    width: int = 256,
    height: int = 256,
    channels: int = 3,
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

    r.append(
        Conv2D(
            4,
            channels,
            padding='same',
            activation='relu',
            use_bias=False,
        )
    )

    r.append(
        MaxPooling2D(
            (2, 2)
        )
    )

    return r


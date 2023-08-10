#/usr/bin/python
from keras.layers   \
    import          \
    Conv2D,         \
    MaxPooling2D,   \
    Rescaling


def input_layer(
    width: int = 32,
    height: int = 32,
    channels: int = 3,
    scale_by: float = 1.0/255
) -> list:
    r: list = list()

    r.append(
        Rescaling(
            scale_by,
            input_shape=(
                height,
                width,
                channels
            )
        )
    )

    r.append(
        Conv2D(
            64,
            channels,
            padding='same',
            activation='relu'
        )
    )

    r.append(
        MaxPooling2D(
            (2, 2)
        )
    )

    return r


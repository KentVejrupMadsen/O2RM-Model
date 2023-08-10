#/usr/bin/python
from keras.layers   \
    import          \
    Conv2D,         \
    MaxPooling2D


def middle_layers(
    channels: int
) -> list:
    r: list = list()

    for l in middle_layer_0ed(
        channels
    ):
        r.append(l)

    for l in middle_layer_1st(
        channels
    ):
        r.append(l)

    for l in middle_layer_2nd(
         channels
    ):
        r.append(l)

    for l in middle_layer_3rd(
         channels
    ):
        r.append(l)

    for l in middle_layer_4th(
         channels
    ):
        r.append(l)

    return r


def middle_layer_0ed(
    channels: int
) -> list:
    r: list = list()

    for i in range(4):
        r.append(
            Conv2D(
                16,
                channels,
                padding='same',
                activation='relu'
            )
        )

    r.append(
        MaxPooling2D(
            2,
            2
        )
    )

    return r


def middle_layer_1st(
    channels: int
) -> list:
    r: list = list()

    for i in range(4):
        r.append(
            Conv2D(
                32,
                channels,
                padding='same',
                activation='relu'
            )
        )

    r.append(
        MaxPooling2D(
            2,
            2
        )
    )

    return r


def middle_layer_2nd(
    channels: int
) -> list:
    r: list = list()

    for i in range(4):
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
            2,
            2
        )
    )

    return r


def middle_layer_3rd(
    channels: int
) -> list:
    r: list = list()

    for i in range(4):
        r.append(
            Conv2D(
                128,
                channels,
                padding='same',
                activation='relu'
            )
        )

    r.append(
        MaxPooling2D(
            2,
            2
        )
    )

    return r


def middle_layer_4th(
    channels: int
) -> list:
    r: list = list()

    for i in range(2):
        r.append(
            Conv2D(
                256,
                channels,
                padding='same',
                activation='relu'
            )
        )

    r.append(
        MaxPooling2D(
            2,
            2
        )
    )

    return r

#/usr/bin/python
from keras.layers   \
    import          \
    Dense,          \
    Flatten,        \
    BatchNormalization


def output_layer(
    classes: int,
    variation: int
) -> list:
    r: list = list()

    r.append(
        BatchNormalization()
    )

    r.append(
        Flatten()
    )

    r.append(
        Dense(
            variation
        )
    )

    r.append(
        Dense(
            variation
            /
            2
        )
    )

    r.append(
        Dense(
            classes
        )
    )

    return r


#/usr/bin/python
from keras.layers   \
    import          \
    Dense,          \
    Flatten


def output_layer(
    classes: int = 10,
    variation: int = 256
) -> list:
    r: list = list()

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


#/usr/bin/python
from keras.models \
    import Sequential

from setup.input \
    import input_layer

from setup.middle \
    import middle_layers

from setup.output \
    import output_layer


def create_new_model(
    width: int = 512,
    height: int = 512,
    channels: int = 3,
    number_of_categories: int = 10,
    variation: int = 1024
):
    model_layers: list = list()

    for layer in input_layer(
        width=width,
        height=height,
        channels=channels
    ):
        model_layers.append(
            layer
        )

    for layer in middle_layers(
        channels=channels
    ):
        model_layers.append(
            layer
        )

    for layer in output_layer(
        classes=number_of_categories,
        variation=variation
    ):
        model_layers.append(
            layer
        )

    model = Sequential(
        model_layers
    )

    return model


create_new_model().summary()

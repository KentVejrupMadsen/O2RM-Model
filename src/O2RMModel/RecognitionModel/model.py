from keras.models \
    import Sequential \
    as KerasModel

from keras.losses \
    import SparseCategoricalCrossentropy

from keras.optimizers.schedules \
    import ExponentialDecay

from keras.optimizers \
    import SGD

from O2RMModel.RecognitionModel.Setup       \
    import                                  \
    input_layer,                            \
    middle_layers,                          \
    output_layer

from random \
    import SystemRandom

class Model(
    KerasModel
):
    def __init__(
        self,
        width: int,
        height: int,
        channels: int,

        categories: int
    ):
        self.width: int = width
        self.height: int = height
        self.channels: int = channels

        self.variation: int = 1024
        self.categories: int = categories

        super().__init__(
            self.make_layers()
        )

        self.compilation()

    def make_layers(self):
        model_layers: list = list()

        for layer in input_layer(
                width=self.width,
                height=self.height,
                channels=self.channels
        ):
            model_layers.append(
                layer
            )

        for layer in middle_layers(
                channels=self.channels
        ):
            model_layers.append(
                layer
            )

        for layer in output_layer(
                classes=self.categories,
                variation=self.variation
        ):
            model_layers.append(
                layer
            )

        return model_layers

    def compilation(self):
        self.compile(
            optimizer=SGD(
                learning_rate=0.0055
            ),
            loss=SparseCategoricalCrossentropy(
                from_logits=True
            ),
            metrics=[
                'accuracy'
            ]
        )



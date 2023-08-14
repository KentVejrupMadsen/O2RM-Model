from keras.callbacks \
    import Callback

from O2RMModel.RecognitionModel.Training.momentum_decay \
    import MomentumDecay


class MomentumDecayCallback(
    Callback
):
    def __init__(
            self, 
            decay=MomentumDecay(),
            measure_momentum_by: str = 'loss'
    ):
        self.decay = decay
        self.measure = measure_momentum_by

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):
        if not(logs is None):
            self.decay.update(
                input_list=logs[
                    self.measure
                ]
            )


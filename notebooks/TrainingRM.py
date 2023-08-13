from os.path \
    import isdir

from random \
    import SystemRandom

from keras.utils \
    import image_dataset_from_directory

import wandb

from keras.backend              \
    import clear_session

from wandb.keras                \
    import WandbMetricsLogger

from keras.models               \
    import Sequential

from keras.layers           \
    import                  \
    RandomFlip,             \
    RandomZoom,             \
    RandomContrast,         \
    RandomBrightness,       \
    RandomRotation

from tensorflow.python.ops  \
  import summary_ops_v2

import tensorflow

from O2RMModel.RecognitionModel.model   \
    import Model                        \
    as RecognitionModel

########################################################################################################################

physical_devices = tensorflow.config.list_physical_devices(
    str(
        'gpu'
    ).upper()
)

selected_physical_device = physical_devices[0]

tensorflow.config.experimental.set_memory_growth(
    selected_physical_device,
    True
)


def generate_seed():
    return SystemRandom().randint(
        1,
        32767
    )


device_name = str('/') + str(
    selected_physical_device.name[
        len('/physical_device:')
        :
        len(selected_physical_device.name)
    ]
)

print(
    'selected: ',
    device_name
)

with tensorflow.device(device_name):
    configuration: dict = {
        'vision': {
            'width': 512,
            'height': 512,
            'channels': 3
        },
        'dataset': {
            'number of labels': 10
        }
    }

    width: int = configuration['vision']['width']
    height: int = configuration['vision']['height']
    channels: int = configuration['vision']['channels']

    number_of_labels: int = configuration['dataset']['number of labels']

    batches: int = 58
    epochs: int = 10

    wandb.init(
        entity='designermadsen',
        project='O2RM',
        config=configuration,
        tags=[
            'Nvidia',
            'Linux',
            'Ubuntu',
            'Development',
            'B&W',
            'test-driven',
            'Bare-Metal',
            'TensorFlow'
        ],
        job_type='Training'
    )

    # Setup of model
    model = RecognitionModel(
        width=width,
        height=height,
        channels=channels,

        categories=number_of_labels
    )

########################################################################################################################
    location_of_model: str = '/opt/models/O2RM'

    if isdir(
        location_of_model
    ):
        model.load_weights(
            location_of_model
        )

########################################################################################################################
    location_of_dataset: str = '/opt/dataset/numbers'

    training_dataset, validation_dataset = image_dataset_from_directory(
        location_of_dataset,
        validation_split=0.20,
        subset='both',
        seed=generate_seed(),
        image_size=(
            height,
            width
        ),
        batch_size=batches
    )

    autotune = tensorflow.data.AUTOTUNE

    wandb.log({
        'training': training_dataset.class_names,
        'validation': validation_dataset.class_names
    })

########################################################################################################################
    def callbacks() -> list:
        callback_list: list = list()

        callback_list.append(
            WandbMetricsLogger()
        )

        return callback_list


    history = model.fit(
        training_dataset.prefetch(
            buffer_size=autotune
        ),
        validation_data=validation_dataset.prefetch(
            buffer_size=autotune
        ),
        epochs=epochs,

        callbacks=callbacks(),

        use_multiprocessing=True,
        workers=4
    )

########################################################################################################################
    model.save(
        location_of_model,
        save_format='tf',
        overwrite=True
    )

    saved_model = wandb.Artifact(
        "o2rm_model",
        type="model"
    )

    saved_model.add_dir(
        location_of_model
    )

    wandb.log_artifact(
        saved_model
    )

    clear_session()

########################################################################################################################
wandb.finish()
summary_ops_v2.flush()

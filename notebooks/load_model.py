from os import walk
from os.path import join
from random import SystemRandom

from keras.models import load_model, Sequential
from keras.utils import img_to_array, load_img
from tensorflow import expand_dims
from tensorflow import nn
from numpy import max, argmax

import wandb


wandb.init(
        entity='designermadsen',
        project='O2RM',
        config={},
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
        job_type='prediction'
    )

random: SystemRandom = SystemRandom()

model: Sequential = load_model('/opt/programming/ORM-Model/resources/Model')
model.summary(
    show_trainable=True,
    expand_nested=True
)

found_images: list = []

for root, directories, files in walk('/opt/dataset/numbers', topdown=False):
    for file in files:
        full_path: str = join(root, file)

        found_images.append(
            full_path
        )

selected_images: list = []

size_of_found_images: int = len(found_images)
last_index_of_found_images: int = size_of_found_images - 1

for i in range(1001):
    selected_images.append(
        found_images[
            random.randint(
                0,
                last_index_of_found_images
            )
        ]
    )

classes = [
    '0.zero',
    '1.one',
    '2.two',
    '3.three',
    '4.four',
    '5.five',
    '6.six',
    '7.seven',
    '8.eight',
    '9.nine'
]


for image_path in selected_images:
    image = load_img(
        image_path
    )

    image_array = img_to_array(
        image
    )

    image_array = expand_dims(
        image_array,
        0
    )

    predictions = model.predict(
        image_array,
        0
    )

    score: float = 100 * max(
        nn.softmax(
            predictions[0]
        )
    )

    label: str = classes[
        argmax(score)
    ]

    log_image = wandb.Image(
        image_array,
        caption=str(label),
    )

    wandb.log(
        {
            'label': label,
            'score': score,
            'original': log_image
        }
    )

wandb.finish()

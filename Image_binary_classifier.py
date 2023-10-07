import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, warnings
import logging
from tensorflow import keras

print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")
tf.get_logger().setLevel(logging.WARNING)


ds_train_ = tf.keras.utils.image_dataset_from_directory(
    'horse-or-human/train',
    labels='inferred',
    label_mode='binary',
    image_size=[300, 300],
    interpolation='nearest',
    batch_size=32,
    shuffle=True,
)

ds_valid_ = tf.keras.utils.image_dataset_from_directory(
    'horse-or-human/validation',
    labels='inferred',
    label_mode='binary',
    image_size=[300, 300],
    interpolation='nearest',
    batch_size=32,
    shuffle=False,
)


def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .prefetch(buffer_size=AUTOTUNE)
    .cache()
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .prefetch(buffer_size=AUTOTUNE)
    .cache()
)

vgg_model = tf.keras.applications.vgg16.VGG16(input_shape=(300, 300, 3), include_top=False)
vgg_model.trainable = False

model = keras.Sequential([
    vgg_model,

    # Head
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

optimizer = tf.keras.optimizers.SGD()
model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=50,
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
plt.show()
print(history_frame)

# model.save('saved_models/vgg16_horse_or_human_SGD')

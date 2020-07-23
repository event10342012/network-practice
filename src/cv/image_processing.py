import os

import tensorflow as tf

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_name = 'fruits-360'
data_dir = os.path.join(root, 'data', data_name)
train_dir = os.path.join(data_dir, 'Training')
val_dir = os.path.join(data_dir, 'Test')

img_width = 100
img_height = 100

train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255).flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=32,
    shuffle=True,
    class_mode='categorical'
)

val_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255).flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    color_mode='rgb',
    batch_size=128,
    shuffle=True,
    class_mode='categorical'
)


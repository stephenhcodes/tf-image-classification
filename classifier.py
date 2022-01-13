from create_model import model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

img_size = (180, 180)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '<put directory reference here>',
    validation_split=0.2,
    subset='training',
    seed=1337,
    image_size=img_size,
    batch_size=batch_size,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    '<put directory reference here>',
    validation_split=0.2,
    subset='validation',
    seed=1337,
    image_size=img_size,
    batch_size=batch_size,
)

augment_data = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip('horizontal'),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = augment_data(images)

augmented_train_ds = train_ds.map(lambda x, y: (augment_data(x, training=True), y))

train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

model = create_model.model(input_shape=img_size + (3,), num_classes=2)

img = keras.preprocessing.image.load_img(
    '<reference to image>', target_size=img_size
)

img_array = tf.expand_dims(keras.preprocessing.image.img_to_array(img), 0)

predictions = model.predict(img_array)
score = predictions[0]
print(
    'This image is %.2f percent a cat and %.2f percent a dog.'
    % (100 * (1 - score), 100 * score)
)

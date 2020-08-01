import os

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory


class FullyConnected(tf.keras.layers.Layer):
    def __init__(self, units=256, initializer='glorot_uniform'):
        super().__init__()
        self.units = units
        self.initializer = initializer

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )

    def call(self, inputs, training=False, **kwargs):
        z = tf.matmul(inputs, self.w) + self.b
        a = tf.nn.relu(z)
        if training:
            a = tf.nn.dropout(a, rate=0.5)
        return a


class Softmax(FullyConnected):
    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.w) + self.b
        return tf.nn.softmax(z)


class Conv2d(tf.keras.layers.Layer):
    def __init__(self,
                 kernel_size,
                 filters,
                 strides,
                 padding='VALID',
                 initializer='glorot_uniform'):
        super(Conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.initializer = initializer

    def build(self, input_shape):
        kernel_size = self.kernel_size + (input_shape[-1], self.filters)
        self.kernel = self.add_weight(
            shape=kernel_size,
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )

        self.bias = self.add_weight(
            shape=(self.filters,),
            initializer=self.initializer,
            trainable=True,
            dtype=self.dtype
        )

    def call(self, inputs, **kwargs):
        output = tf.nn.conv2d(
            input=inputs,
            filters=self.kernel,
            strides=self.strides,
            padding=self.padding
        )

        output = tf.nn.bias_add(output, self.bias)
        return tf.nn.relu(output)


class AlexNet(tf.keras.Model):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
        self.conv1 = Conv2d(kernel_size=(11, 11), filters=96, strides=4, padding='VALID')
        self.conv2 = Conv2d(kernel_size=(5, 5), filters=256, strides=1, padding='SAME')
        self.conv3 = Conv2d(kernel_size=(3, 3), filters=384, strides=1, padding='SAME')
        self.conv4 = Conv2d(kernel_size=(3, 3), filters=384, strides=1, padding='SAME')
        self.conv5 = Conv2d(kernel_size=(3, 3), filters=256, strides=1, padding='SAME')
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = FullyConnected()
        self.fc2 = FullyConnected()
        self.softmax = Softmax(units=2)

    def call(self, inputs, training=False, **kwargs):
        a = self.normalization_layer(inputs)
        a = self.conv1(a)
        a = tf.nn.max_pool(a, ksize=(3, 3), strides=2, padding='VALID')
        a = tf.nn.lrn(a, alpha=10 ** -4, beta=0.75)
        a = self.conv2(a)
        a = tf.nn.max_pool(a, ksize=(2, 2), strides=2, padding='VALID')
        a = tf.nn.lrn(a, alpha=10 ** -4, beta=0.75)
        a = self.conv3(a)
        a = self.conv4(a)
        a = self.conv5(a)
        a = tf.nn.max_pool(a, ksize=(3, 3), strides=2, padding='VALID')
        a = self.flatten(a)
        a = self.fc1(a, training=training)
        a = self.fc2(a, training=training)
        a = self.softmax(a)
        return a


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images)
    loss = loss_object(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)


if __name__ == '__main__':
    data_name = 'dogs_vs_cats'
    epochs = 10
    batch_size = 32
    image_size = (256, 256)

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(root, 'data', data_name)
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # load data
    train_ds = image_dataset_from_directory(train_dir,
                                            image_size=image_size,
                                            batch_size=batch_size)
    test_ds = image_dataset_from_directory(test_dir,
                                           image_size=image_size,
                                           batch_size=batch_size)

    # instantiate model
    model = AlexNet()

    # instantiate loss object
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # instantiate optimizer
    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # start training
    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for step, (images, labels) in enumerate(train_ds):
            train_step(images, labels)
            if step % 100 == 0:
                template = 'Epoch: {}, Step {}, Loss: {}, Accuracy: {}'
                print(template.format(epoch + 1,
                                      step,
                                      train_loss.result() * 100,
                                      train_accuracy.result() * 100))

        for (images, labels) in test_ds:
            test_step(images, labels)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result() * 100,
                              train_accuracy.result() * 100,
                              test_loss.result() * 100,
                              test_accuracy.result() * 100))

    model.save(filepath=os.path.join(root, 'ckpt', 'alexnet'))

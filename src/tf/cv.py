import tensorflow as tf


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
    def call(self, inputs, training=False, **kwargs):
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
        self.conv1 = Conv2d(kernel_size=(5, 5), filters=96, strides=2, padding='SAME')
        self.conv2 = Conv2d(kernel_size=(5, 5), filters=256, strides=1, padding='SAME')
        self.conv3 = Conv2d(kernel_size=(3, 3), filters=384, strides=1, padding='SAME')
        self.conv4 = Conv2d(kernel_size=(3, 3), filters=384, strides=1, padding='SAME')
        self.conv5 = Conv2d(kernel_size=(3, 3), filters=256, strides=1, padding='SAME')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = FullyConnected()
        self.dense2 = FullyConnected()
        self.softmax = Softmax(units=10)

    def call(self, inputs, training=False, **kwargs):
        a = self.conv1(inputs)
        a = tf.nn.max_pool(a, ksize=(2, 2), strides=2, padding='VALID')
        a = self.conv2(a)
        a = tf.nn.max_pool(a, ksize=(2, 2), strides=2, padding='VALID')
        a = self.conv3(a)
        a = self.conv4(a)
        a = self.conv5(a)
        a = tf.nn.max_pool(a, ksize=(2, 2), strides=2, padding='VALID')
        a = self.flatten(a)
        a = self.dense1(a)
        a = self.dense2(a)
        a = self.softmax(a)
        return a


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    model = AlexNet()
    model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
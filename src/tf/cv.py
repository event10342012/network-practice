import tensorflow as tf


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
            trainable=True
        )

        self.bias = self.add_weight(
            shape=(self.filters,),
            initializer=self.initializer,
            trainable=True
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


class AlexNet(tf.keras.layers.Layer):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = Conv2d(kernel_size=(11, 11), filters=96, strides=4)
        self.conv2 = Conv2d(kernel_size=(5, 5), filters=256, strides=1)
        self.conv3 = Conv2d(kernel_size=(3, 3), filters=384, strides=1)
        self.conv4 = Conv2d(kernel_size=(3, 3), filters=384, strides=1)
        self.conv5 = Conv2d(kernel_size=(3, 3), filters=256, strides=1)

    def call(self, inputs, **kwargs):
        a = self.conv1(inputs)
        a = tf.nn.max_pool(input=a, ksize=(3, 3), strides=2, padding='VALID')
        return a


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255
    x_test = x_test / 255

    x = tf.ones(shape=(10, 227, 227, 3), dtype=tf.float32)
    layer = AlexNet()
    output = layer(x)
    print(output.shape)

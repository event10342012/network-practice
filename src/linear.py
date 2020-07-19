import tensorflow as tf
from tensorflow import keras


class Linear(keras.layers.Layer):
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            name='w'
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            name='b'
        )

    def call(self, inputs, **kwargs):
        return tf.matmul(inputs, self.w) + self.b


class MLP(keras.layers.Layer):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = Linear()
        self.l2 = Linear()
        self.l3 = Linear()

    def call(self, inputs, **kwargs):
        x = tf.nn.relu(self.l1(inputs))
        x = tf.nn.relu(self.l2(x))
        x = self.l3(x)
        return x


if __name__ == '__main__':
    mlp = MLP()
    y = mlp(tf.ones((2, 3)))
    print(y.shape)

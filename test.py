import tensorflow as tf


x1 = tf.constant(1.)
x2 = tf.Variable(1, trainable=True, dtype=tf.float32)
x2.assign_add(3)
y = x1 + x2
print(x2)

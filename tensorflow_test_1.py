from __future__ import print_function


import tensorflow as tf


# hello = tf.constant("Hello, Tensorflow on Windows!")
# sess = tf.Session()
# print(sess.run(hello))

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)  # also tf.float32 implicitly
# print(node1, node2)

node3 = tf.add(node1, node2)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b  # + provides a shortcut for tf.add(a, b)

add_and_triple = adder_node * 3.

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

# Create the session to build project
sess = tf.Session()

# Init is needed to initialize all tf.Variable because Constants don't need this
init = tf.global_variables_initializer()
sess.run(init)


# print(sess.run([node1, node2]))
# print("node3: ", node3)
# print("sess.run(node3): ", sess.run(node3))

# print(sess.run(adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

# print(sess.run(add_and_triple, {a: 3, b: 4.5}))

# by using a bracket of numbers for x I can test multiple numbers at once
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

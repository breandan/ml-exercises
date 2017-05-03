import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x_train = np.linspace(-1, 1, 100)
y_train = 2 * x_train + np.random.randn(100) * 0.25

weight = tf.Variable(0.0)
b = tf.Variable(0.0)

X = tf.placeholder("float")
Y = tf.placeholder("float")

y_model = tf.add(tf.multiply(X, weight), b)
loss = tf.square(Y - y_model)
training_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(100):
    for (x, y) in zip(x_train, y_train):
        sess.run(training_op, feed_dict={X: x, Y: y})
    prediction = sess.run(weight)
    # print(prediction)
    print("Slope: " + str(weight.eval(sess)) + ", B: "+ str(b.eval(sess)))
    print()

sess.close()

y_learned = x_train * prediction

plt.plot(x_train, y_learned, 'r')
plt.scatter(x_train, y_train)
plt.show()
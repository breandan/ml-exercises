import tensorflow as tf

c = tf.constant(5.0)
u = tf.constant(7.0)
v = tf.Variable(0.0)

saver = tf.train.Saver()
cu = tf.mul(c, u)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
v = tf.mul(c, u)
print(sess.run(v))

saver.save(sess, "/tmp/model.ckpt")
sess.close()


sess = tf.Session()
saver.restore(sess, "/tmp/model.ckpt")
vars = tf.trainable_variables()
for v in vars:
    print(v.name)

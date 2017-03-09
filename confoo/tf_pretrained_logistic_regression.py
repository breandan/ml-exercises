import tensorflow as tf
import numpy as np

from PIL import Image

filename_queue = tf.train.string_input_producer(
    ['digit.png'])  # list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

saver = tf.train.Saver()

def load_image( infilename ) :
    img = Image.open( infilename )
    img = img.convert('1')  # convert image to black and white
    img.load()
    data = np.asarray( img, dtype="float" )
    return data

with tf.Session() as sess:
    saver.restore(sess, "confoo/pretrained/model.ckpt")
    print("Model restored from file")
    images = np.zeros((1, 784))

    pngFile = load_image("confoo/digit.png")

    """
    all images in the training set have an range from 0-1
    and not from 0-255 so we divide our flatten images
    (a one dimensional vector with our 784 pixels)
    to use the same 0-1 based range
    """

    flatten = pngFile.flatten()

    """
    we need to store the flatten image and generate
    the correct_vals array
    correct_val for a digit (9) would be
    [0,0,0,0,0,0,0,0,0,1]
    """

    images[0] = flatten

    np.set_printoptions(suppress=True)
    print("           0          1            2            3           4            5            6            7          8           9 ")
    print(np.array_repr(sess.run(pred, feed_dict={x: images})[0]).replace('\n', '').replace(' ', '').replace(',', ', '))
    my_classification = sess.run(tf.argmax(pred, 1), feed_dict={x: images})


    """
    we want to run the prediction and the accuracy function
    using our generated arrays (images and correct_vals)
    """
    print('logistic regression predicted', my_classification[0], "for your digit")



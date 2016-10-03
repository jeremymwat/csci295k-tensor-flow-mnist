import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 50

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


input_imgs = tf.placeholder(tf.float32, [None, 784])
labels = tf.placeholder(tf.float32, [None, 10])

###
# Initialize weights and biases
###


weight_one_convolution = tf.Variable(tf.truncated_normal([5,5,1,32], stddev=0.1))
bias_one_convolution = tf.Variable(tf.zeros([32]))

weight_two_convolution = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1))
bias_two_convolution = tf.Variable(tf.zeros([64]))

weight_one_ff = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
bias_one_ff = tf.Variable(tf.zeros([1024]))

weight_two_ff = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1))
bias_two_ff = tf.Variable(tf.zeros([10]))


### Create forward operation
input_img_reshape = tf.reshape(input_imgs, [-1,28,28,1])

first_conv = tf.nn.conv2d(input_img_reshape, weight_one_convolution, strides=[1, 1, 1, 1], padding='SAME')
first_relu = tf.nn.relu(first_conv+bias_one_convolution)
max_pool_1 = tf.nn.max_pool(first_relu, [1,2,2,1], [1,2,2,1], 'SAME')

second_conv = tf.nn.conv2d(max_pool_1, weight_two_convolution, strides=[1, 1, 1, 1], padding='SAME')
second_relu = tf.nn.relu(second_conv+bias_two_convolution)
max_pool_2 = tf.nn.max_pool(second_relu, [1,2,2,1], [1,2,2,1], 'SAME')


reshaped = tf.reshape(max_pool_2, [-1, 7*7*64])
forward_one = tf.nn.relu(tf.matmul(reshaped, weight_one_ff)+bias_one_ff)

dropout_placeholder = tf.placeholder(tf.float32)
forward_with_dropout = tf.nn.dropout(forward_one, dropout_placeholder)

forward_two = tf.nn.softmax(tf.matmul(forward_with_dropout, weight_two_ff)+bias_two_ff)

######

cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(forward_two), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

init = tf.initialize_all_variables()
correct_prediction = tf.equal(tf.argmax(forward_two,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(init)



for i in range(2000):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  sess.run(train_step, feed_dict={input_imgs: batch_xs, labels: batch_ys, dropout_placeholder: 0.5})


print(sess.run(accuracy, feed_dict={input_imgs: mnist.test.images, labels: mnist.test.labels, dropout_placeholder: 1.0}))

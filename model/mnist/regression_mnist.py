import tensorflow as tf 

class Regression_mnist(object):
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.w = tf.Variable(tf.zeros([784,10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.w)+ self.b)
        self.y_ = tf.placeholder(tf.float32, [None,10])
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))

 
    def train_mnist_regression(self, mnist):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step = tf.train.GradientDescentOptimizer(0.005).minimize(self.cross_entropy)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(1000):
                sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
            
            print 'iters:', step+1
        
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            return sess.run(accuracy, feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})


if __name__ == '__main__':
    train = Regression_mnist()
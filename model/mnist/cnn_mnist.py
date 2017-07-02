import tensorflow as tf 
import prettytensor as pt


class CNN_mnist(object):
    def __init__(self, image_size=(28,28,1), lr_rate=5e-5):
        self.height, self.width, self.channel = image_size
        self.nClasses = 10

        self.x = tf.placeholder("float", [None, 784])
        self.y_ = tf.placeholder("float", [None, 10])

        self.w = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))

    def train_model(self, mnist):
        self.images = tf.placeholder(tf.float32, shape=(None, self.height, self.width, self.channel))
        
        y = tf.nn.softmax(tf.matmul(self.x,self.w) + self.b)
        cross_entropy = -tf.reduce_sum(self.y_*tf.log(y))

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        for i in range(1000):
          batch = mnist.train.next_batch(50)
          train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1]})
        
        print "iters: ", i+1
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        return accuracy.eval(feed_dict={self.x: mnist.test.images, self.y_: mnist.test.labels})

if __name__ == '__main__':
    CNN_mnist()

from lib import data
from model.mnist import regression_mnist
from model.mnist import cnn_mnist
from model.mnist import cnn_model


class Test(object):
    def __init__(self):
        self.mnist = data.load_mnist()


    def test_regression_mnist(self):
        train = regression_mnist.Regression_mnist()
        results = train.train_mnist_regression(self.mnist)

        print results


    def test_cnn_mnist(self):
        train = cnn_mnist.CNN_mnist()
        results = train.train_model(self.mnist)

        print results

    def test_cnn_model(self):
        train = cnn_model.Model()
        results = train.train_model(self.mnist)


if __name__ == '__main__':
    test = Test()
    test.test_regression_mnist()
    test.test_cnn_mnist()
    test.test_cnn_model()

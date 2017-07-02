from tensorflow.examples.tutorials.mnist import input_data


def load_mnist():
    mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
    return mnist 


if __name__ == '__main__':
    load_mnist()
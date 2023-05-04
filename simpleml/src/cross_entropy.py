import gzip
import numpy as np

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    with gzip.open(image_filename, "rb") as f:
        X = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    with gzip.open(label_filename, "rb") as f:
        y = np.frombuffer(f.read(), np.uint8, offset=8)
    return X, y

def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    exp_sum_z = np.sum(np.exp(Z), axis=-1)
    b = Z.shape[0]
    z_y = Z[np.arange(b), y]
    loss = np.mean(np.log(exp_sum_z) - z_y)

    return loss


def softmax(x):
    x1 = np.exp(x)
    return x1 / np.sum(x1, axis=-1, keepdims=True)

def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    n = X.shape[0]
    step = n // batch
    index = np.arange(batch)
    for i in range(step + 1):
        start = i * batch
        end = min(start + batch, n)
        if start == end:
            break
        x1 = X[start: end]
        y1 = y[start: end]
        z = softmax(np.matmul(x1, theta))
        z[index, y1] -= 1
        grad = np.matmul(x1.transpose(), z) / batch
        theta -= lr * grad

def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)

def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("../data/train-images-idx3-ubyte.gz",
                             "../data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("../data/t10k-images-idx3-ubyte.gz",
                             "../data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=100, lr = 0.1)
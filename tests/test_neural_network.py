"""
Tests directly related to the class _NN and it's children.

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TODO relative imports
from aglaia import SPNN
from aglaia.aglaia import _NN
from aglaia.wrappers import _OSPNN, OSPSPNN
from aglaia.utils import InputError


def hidden_layer_sizes(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hidden_layer_sizes = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hidden_layer_sizes = [4,5])
    C(hidden_layer_sizes = (4,5))
    C(hidden_layer_sizes = [4.0])

    # This should be caught
    catch([])
    catch([0,4])
    catch([4.2])
    catch(["x"])
    catch([None])
    catch(None)
    catch(4)
    catch([0])

def l1_reg(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(l1_reg = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(l1_reg = 0.1)
    C(l1_reg = 0.0)

    # This should be caught
    catch(-0.1)
    catch("x")
    catch(None)
    catch([0])

def l2_reg(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(l2_reg = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(l2_reg = 0.1)
    C(l2_reg = 0.0)

    # This should be caught
    catch(-0.1)
    catch("x")
    catch(None)
    catch([0])

def batch_size(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(batch_size = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(batch_size = 2)
    C(batch_size = 2.0)
    C(batch_size = "auto")

    # This should be caught
    catch(1)
    catch(-2)
    catch("x")
    catch(4.2)
    catch(None)

def learning_rate(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(learning_rate = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(learning_rate = 0.1)

    # This should be caught
    catch(0.0)
    catch(-0.1)
    catch("x")
    catch(None)

def iterations(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(iterations = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(iterations = 1)
    C(iterations = 1.0)

    # This should be caught
    catch(-2)
    catch("x")
    catch(4.2)
    catch(None)

def tf_dtype(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(tf_dtype = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(tf_dtype = "64")
    C(tf_dtype = 64)
    C(tf_dtype = "float64")
    C(tf_dtype = tf.float64)
    C(tf_dtype = "32")
    C(tf_dtype = 32)
    C(tf_dtype = "float32")
    C(tf_dtype = tf.float32)
    C(tf_dtype = "16")
    C(tf_dtype = 16)
    C(tf_dtype = "float16")
    C(tf_dtype = tf.float16)

    # This should be caught
    catch(8)
    catch("x")
    catch(float)
    catch(None)

def hl1(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl1 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl1 = 1)
    C(hl1 = 1.0)

    # This should be caught
    catch(0)
    catch("x")
    catch(4.2)
    catch(None)
    catch(-1)

def hl2(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl2 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl2 = 1)
    C(hl2 = 1.0)
    C(hl2 = 0)

    # This should be caught
    catch("x")
    catch(4.2)
    catch(None)
    catch(-1)

def hl3(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(hl2 = 2, hl3 = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(hl2 = 2, hl3 = 1)
    C(hl2 = 2, hl3 = 1.0)
    C(hl2 = 2, hl3 = 0)

    # This should be caught
    catch("x")
    catch(4.2)
    catch(None)
    catch(-1)

def representation(C):
    # Exceptions that are supposed to be caught
    def catch(s):
        try:
            C(representation = s)
            raise Exception
        except InputError:
            pass

    # This should not raise an exception
    C(representation = "unsorted_couLomb_matrix")
    C(representation = "sorted_couLomb_matrix")
    C(representation = "bag_of_bOnds")
    C(representation = "slAtm")

    # This should be caught
    catch("none")
    catch(4.2)
    catch(None)
    catch(-1)

def test_input():
    # Additional test that inheritance is ok
    for C in _NN, SPNN, _OSPNN, OSPSPNN:
        hidden_layer_sizes(C)
        l1_reg(C)
        l2_reg(C)
        batch_size(C)
        learning_rate(C)
        iterations(C)
        tf_dtype(C)

    for C in _OSPNN, OSPSPNN:
        hl1(C)
        hl2(C)
        hl3(C)

    representation(OSPSPNN)


def test_mrmp():
    # set matplotlib to be interactive so the plots wont show
    # TODO should probably set agg backend instead
    plt.ion()

    # Simple example of fitting a quadratic function
    estimator = SPNN(hidden_layer_sizes=(5, 5, 5), learning_rate=0.01, iterations=5000, l2_reg = 0, tf_dtype = 32, scoring_function="rmse")
    x = np.arange(-2.0, 2.0, 0.05)
    X = np.reshape(x, (len(x), 1))
    y = np.reshape(X ** 3, (len(x), 1))

    estimator.fit(X, y)
    y_pred = estimator.predict(X)

    # Cost plot
    estimator.plot_cost()
    estimator.correlation_plot(y_pred, y)

if __name__ == "__main__":
    test_input()
    test_mrmp()

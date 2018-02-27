"""
Tests for the Osprey wrapper.
"""

import tensorflow as tf
import numpy as np

# TODO relative imports
from .aglaia import _NN, MRMP
from .wrappers import _OSPNN, OSPMRMP
from .utils import InputError
from sklearn.base import clone

def test_cloning():

    estimator = OSPMRMP(batch_size=100, representation="unsorted_coulomb_matrix",
                                 hl1=12, optimiser=tf.train.AdadeltaOptimizer, scoring_function='mae',
                                 iterations=6000, slatm_sigma1=0.1, activation_function=tf.nn.relu)

    dolly = clone(estimator)

    params_estimator = estimator.get_params()
    params_dolly = dolly.get_params()

    assert len(params_estimator) == len(params_dolly)

    assert params_estimator['iterations'] == params_dolly['iterations']
    assert params_estimator['hl1'] == params_dolly['hl1']
    assert params_estimator['l2_reg'] == params_dolly['l2_reg']
    
    np.testing.assert_array_almost_equal(estimator.descriptor, dolly.descriptor, decimal=6)
    np.testing.assert_array_almost_equal(estimator.properties, dolly.properties, decimal=6)
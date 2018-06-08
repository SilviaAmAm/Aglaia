"""
This example shows how to use the ARMP_G estimator from Aglaia. It shows how to create the estimator, how to train it,
how to score the performance of it and how to use it to make predictions.
"""

from aglaia import aglaia
import numpy as np
from sklearn import model_selection as modsel

## ------------- ** Loading the data ** ---------------

npzfile = np.load("../data/CN_isopentane_forces.npz")
coordinates = npzfile['arr_0']
zs = npzfile['arr_1']
energies = npzfile['arr_2']
forces = npzfile['arr_3']

## ------------- ** Fitting an estimator ** ---------------

estimator = aglaia.ARMP_G(iterations=10)

estimator.fit([coordinates, zs], [energies, forces])
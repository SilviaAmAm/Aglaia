import joblib
import sys
sys.path.insert(0,'/Users/walfits/Repositories/Aglaia/')
import energies_NN as nn
import numpy as np
from sklearn import model_selection as modsel
from multiprocessing import Pool
import contextlib


reg_l1_list = [0.0001, 0.0003, 0.0005, 0.0007, 0.0009]

# Loading the data set
data = joblib.load("cc_kjmol_invdist.bz")
X = data["X"]
y = np.reshape(data["y"], (data["y"].shape[0],))

# Splitting the data
X_train, X_test, y_train, y_test = modsel.train_test_split(X, y, test_size=0.2, random_state=42)


f = open('scores_4_nodes.txt', 'w')

for reg_l1 in reg_l1_list:

    # Training the model
    estimator = nn.Energies_NN(max_iter=20000, learning_rate_init=0.018, hidden_layer_sizes=(4,), batch_size=1000,
                               alpha=0.000001, alpha_l1=reg_l1, tensorboard=False)
    estimator.fit(X_train, y_train)

    score = estimator.scoreFull(X_test, y_test)

    estimator.plotWeights_no_diag(plot_fig=True)

    f.write("The score for the model with reg " + str(reg_l1) + " is: \n")
    f.write(str(score) + "\n")
    f.write("\n")


f.close()










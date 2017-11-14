import joblib
import sys
sys.path.insert(0,'/Users/walfits/Repositories/Aglaia/')
import energies_NN as nn
import numpy as np
from sklearn import model_selection as modsel

# Loading the data
data = joblib.load("cc_demeaned_kjmol.bz")
X = data["X"]
y = np.reshape(data["y"], (data["y"].shape[0],))

# Splitting the data
X_train, X_test, y_train, y_test = modsel.train_test_split(X, y, test_size=0.2, random_state=1)

# Training the model
estimator = nn.Energies_NN(max_iter=1, learning_rate_init=0.005, hidden_layer_sizes=(13,), batch_size=1000,
                              alpha=0.0005, tensorboard=False)
estimator.fit(X_train, y_train)
print(estimator.scoreFull(X_test, y_test))

np.random.seed(seed=1)
random_input = abs(np.random.random_sample(28,))


estimator.vis_input_network(random_input, alpha_l1=0.0001, alpha_l2=0.0, clipping=0)
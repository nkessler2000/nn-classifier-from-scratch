from ds_functions import *
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from time import time

hl_sizes = (200,25)
lam = 3
maxiter = 1000
tol = 1e-6
solver = 'L-BFGS-B' # ‘L-BFGS-B’ 'BFGS'

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y.reshape(-1), test_size=0.1)

my_nn = MyNeuralClassifier(hl_sizes, lam, maxiter, tol, solver)

# fit model
start_time = time()
my_nn.fit(X_tr, y_tr)
print('Fitting completed in {0} seconds'.format(time() - start_time))

# get predictions
pred_tr = my_nn.predict(X_tr)
pred_ts = my_nn.predict(X_ts)
# get score and classification report

score_tr = np.mean((pred_tr == y_tr))
score_ts = np.mean((pred_ts == y_ts))

cr_tr = classification_report(y_tr, pred_tr)
cr_ts = classification_report(y_ts, pred_ts)

print(score_tr)
print(cr_tr)

print(score_ts)
print(cr_ts)

pass

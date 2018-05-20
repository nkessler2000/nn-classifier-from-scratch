from MyNeuralClassifier import *
from ds_functions import *
from sklearn.datasets import fetch_mldata

hl_sizes = (200,50)
lam = 3
maxiter = 500
tol = 1e-5
solver = 'L-BFGS-B'  # ‘CG’ 'TNC' 'L-BFGS-B'

mnist = fetch_mldata('MNIST original')
X = mnist.data
y = mnist.target

X_tr, X_ts, y_tr, y_ts = train_test_split(X, y.reshape(-1), test_size=0.1)
sorted_classes = np.unique(y).tolist()

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
p = cm_plot(confusion_matrix(y_tr, pred_tr), 
            classes=sorted_classes, 
            title='Confusion Matrix - Train Set - MyNeuralClassifier', 
            cmap=plt.cm.Blues, normalize=True, cbar=False)
plt.show()

print(score_ts)
print(cr_ts)

p = cm_plot(confusion_matrix(y_ts, pred_ts), 
            classes=sorted_classes, 
            title='Confusion Matrix - Test Set - MyNeuralClassifier', 
            cmap=plt.cm.Blues, normalize=True, cbar=False)
plt.show()

# my classifier vs sklearn's MLP, using same params
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hl_sizes, max_iter=maxiter, shuffle=False, solver='lbfgs', tol=tol)

start_time = time()
mlp.fit(X_tr, y_tr)
print('Fitting completed in {0} seconds'.format(time() - start_time))

pred_mlp = mlp.predict(X_ts)
print(classification_report(y_ts, pred_mlp))

p = cm_plot(confusion_matrix(y_ts, pred_mlp), 
            classes=sorted_classes, 
            title='Confusion Matrix - Test Set - MLP Classifier',
            cmap=plt.cm.Blues, normalize=True, cbar=False)
plt.show()

# save fitted model to disk
save_pickle(my_nn, 'my_nn.pkl.bz2', bz2)

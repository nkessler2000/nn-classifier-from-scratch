## Neural Network Classifier from Scratch

This code contains an implementation of a neural network classifier built from scratch. 

### Usage
#### Import class `MyNeuralClassifier` from `MyNeuralClassifier.py`

`from MyNeuralClassifier import MyNeuralClassifier`

#### Instantiate the class
`my_nn = MyNeuralClassifier()`<Br>
__Options__<Br>
- `hl_sizes` - A tuple containing hidden layer sizes. _Default: (100,)_
- `lam`- Lambda regularization term. _Default: 0_
- `maxiter`- Maximum number of iterations for the optimizer. _Default: 100_
- `tol` - Tolerance threshold for optimizer. _Default: 1e-5_
- `solver` - Solver method for optimizer. _Default: L-BFGS-B_

#### Fit the data
`my_nn.fit(X,y)`<br>
__Note:__ Data must be stored as numeric numpy arrays.

#### Generate predictions
`my_nn.predict(X)` <br>
Outputs a numpy array containing predicted labels.

#### Other files in repo: 
- `MyNeuralClassifier - MNIST data.ipynb' - Jupyter notebook for testing classifer with MNIST dataset
- `ds_functions.py` - Additional functions
- `__main__.py` - same code as Jupyter notebook
- `requirements.txt` - Install required modules using `pip install -r requirements.txt`
- `my_nn.pkl.bz2` - Pickled trained model

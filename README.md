## Neural Network Classifier from Scratch

This code contains an implementation of a neural network classifier built from scratch. 

### Usage
1. Import class `MyNeuralClassifier` from `MyNeuralClassifier.py`

`from MyNeuralClassifier import MyNeuralClassifier`

2. Instantiate the class
`my_nn = MyNeuralClassifier()
####Options
-`l_sizes`- A tuple containing hidden layer sizes. Default: (100,)
-`am`- Lambda regularization term. Default: 0
-`maxiter`- Maximum number of iterations for the optimizer. Default: 100
-`tol` - Tolerance threshold for optimizer. Default: 1e-5
-`solver` - Solver method for optimizer. Default: L-BFGS-B

3. Fit the data
`my_nn.fit(X,y)`
__Note:__ Data must be stored as numeric numpy arrays.

4. Generate predictions
`my_nn.predict(X)` 
Outputs a numpy array containing predicted labels.

Other files in repo: 
`MyNeuralClassifier - MNIST data.ipynb' - Jupyter notebook for testing classifer with MNIST dataset
`ds_functions.py` - Additional functions
`__main__.py` - same code as Jupyter notebook
`requirements.txt` - Install required modules using `pip install -r requirements.txt`
`my_nn.pkl.bz2` - Pickled trained model

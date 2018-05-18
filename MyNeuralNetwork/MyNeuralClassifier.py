import numpy as np
from scipy.optimize import minimize

class MyNeuralClassifier():
    def __init__(self, hidden_layer_sizes=(100,0), lam=0, maxiter=100, tol=1e-5, solver='L-BFGS-B'):
        self.__maxiter = maxiter
        self.__hl_sizes = hidden_layer_sizes
        self.__lam = lam
        self.__solver = solver
        self.__opt_thetas = None
        self.__tol = tol

    def __init_thetas(self, hl_sizes, n_features, n_classes):
        """Build list of initiali weight vectors"""
        
        def get_weight(layer_in, layer_out, epsilon=0.1):
            """Create a random weight vector Theta of the specified size"""
            weight = np.random.rand(layer_out, layer_in + 1) 
            weight = weight * 2 * epsilon - epsilon
            return weight

        Thetas = []
        # add Theta for input layer
        Thetas.append(get_weight(n_features, hl_sizes[0]))
        # add additional Thetas
        for i in range(1, len(hl_sizes) + 1):
            # in is the number of units in the hidden layer
            l_in = hl_sizes[i-1]
            # out is either the number of units in the next hidden layer, or
            # the number of classes, if this is the last hidden layer
            l_out = n_classes if i == len(hl_sizes) else hl_sizes[i]
            Thetas.append(get_weight(l_in, l_out))
        
        return Thetas
        
    def __unroll_thetas(self, Thetas_flat, hl_sizes, n_classes, n_features):
        """Unrolls flattened Thetas vector and returns list of Theta matrices.
        Parameters are used to determine matrix shapes"""
        n_thetas = len(hl_sizes) + 1 # for 1 hiden layer, 2 Thetas
        ret = []
        start_pos = 0
        for i in range(0, n_thetas):
            rows = hl_sizes[i] if (i+1) < n_thetas else n_classes
            cols = n_features + 1 if i == 0 else hl_sizes[i-1] + 1
            end_pos = start_pos + (rows * cols)
            values = Thetas_flat[start_pos:end_pos] 
            ret.append(np.reshape(values, (rows, cols)))
            start_pos = end_pos
        return ret
    
    def __flatten_arrays(self, arr_list):
        """Takes a list of arrays, flattens and concatenates all
        and returns a single vector"""
        ret = [a.flatten('C') for a in arr_list] # 'C' to flatten row-wise
        ret = np.concatenate(ret, axis=0)
        return ret
        
    def __get_y_matrix(self, y):
        """returns maxtrix representation of target values"""
        y = y.reshape(-1)
        n_classes = len(np.unique(y))
        y_mat = np.zeros((len(y), n_classes))
        for i, val in enumerate(np.unique(y)):
            row = (y == val)
            y_mat[:,i] = row
        return y_mat
    
    def __sigmoid(self, x):
        """Returns sigmoid function of an array"""
        return 1 / (1 + np.exp(-x))
    
    def __sigmoid_prime(self, x):
        """Returns derivative of sigmoid function"""
        return self.__sigmoid(x) * (1 - self.__sigmoid(x))
    
    def __cost_func(self, Thetas_rolled, X, y, hl_sizes, lam, return_grad=True):
        """Computes the cost function value, and also returns gradient matrices"""
        def get_reg_term(Thetas, m, lam=0):
            """Computes the regularization term"""
            t_sums_sq = 0
            for t in Thetas:
                t_sums_sq += np.sum(np.power(t[:, 1:], 2))
            return (lam/(2*m)) * t_sums_sq        
            
        def get_cost(a_out, y_mat, m):
            """Computes cost value"""
            cost = np.sum(-y_mat * np.log(a_out) - (1 - y_mat) * np.log(1-a_out))
            cost = cost/m
            return cost
        
        def get_deltas(Thetas, a_vals, z_vals, y_mat):
            """Computes the error matrices using backwards propagation"""
            deltas = []
            # add in the delta term for the final layer
            deltas.append(a_vals[-1] - y_mat)
            
            # next, add in deltas for ealier layers
            for i in range(1, len(Thetas)):
                sig_prime = self.__sigmoid_prime(z_vals[-(i+1)])
                t = Thetas[-i][:, 1:] # drop bias unit column
                d = np.dot(deltas[i-1], t)
                d = d * sig_prime
                deltas.append(d)
            
            # now, to get our Delta values, we can multiply the deltas by the activation unit values
            Deltas = []
            for i, d in enumerate(deltas):
                a = a_vals[-(i+2)]
                D = np.dot(np.transpose(d), a)
                Deltas.append(D)
        
            return [D for D in reversed(Deltas)]
        
        def get_activation_units(Thetas, X, y_mat, lam):
            """Gets the activation unit values"""
            # number of observations
            m = len(X[:,])
        
            # we next need to compute our activation units
            a_vals = [] 
            z_vals = []
            # remember, a_1 is just X with the bias unit added in
            for i in range(len(Thetas) + 1):
                if i == 0:
                    # this is our input layer. a_1 = X plus bias unit
                    a = np.append(np.ones([len(X),1]), X, axis=1)
                    a_vals.append(a)
                else:
                    a = a_vals[i-1]
                    t = Thetas[i-1]            
                    # get product
                    z = np.dot(a, np.transpose(t))
                    z_vals.append(z)
                    # apply sigmoid function
                    a_next = self.__sigmoid(z)
                    # append a bias unit. Skip if final layer
                    if i != len(Thetas):
                        a_next = np.append(np.ones([len(a_next),1]), a_next, axis=1)
                    # append to list
                    a_vals.append(a_next)
            return a_vals, z_vals
        
        def get_gradients(Deltas, Thetas, m):
            """Computes the gradient values and returns as a 1D array"""
            grads = []
            for i, t in enumerate(Thetas):
                D = Deltas[i]
                t[:,0] = 0 # set bias unit column values to zero
                grad = (D/m) + ((lam/m) * t)
                grads.append(grad)
            ret = self.__flatten_arrays(grads)
            return ret
    
        # get y as a matrix
        y_mat = self.__get_y_matrix(y)
        # number of observations
        m = len(X[:,])    
       
        # unroll thetas 
        Thetas = self.__unroll_thetas(Thetas_rolled, hl_sizes, self.__n_classes, self.__n_features)
        
        # build activation unit values
        a_vals, z_vals = get_activation_units(Thetas, X, y_mat, lam)
    
        # now, with the activation units, we can compute cost
        reg_term = get_reg_term(Thetas, m, lam) if lam != 0 else 0
        cost = get_cost(a_vals[-1], y_mat, m) + reg_term

        # if return_grads is False, exit here
        if not return_grad:
            return cost

        # get the Delta values to compute the gradients
        Deltas = get_deltas(Thetas, a_vals, z_vals, y_mat)
        # now get the gradients
        grad = get_gradients(Deltas, Thetas, m)
        # make sure dimensions match
        if len(grad) != len(Thetas_rolled):
            raise Exception('Length of input vector and gradient vector don\'t match, {0}, {1}'.format(
                len(grad), len(Thetas_rolled)))
        return cost, grad
    
    def fit(self, X, y):
        # build initial theta values
        if len(X) != len(y):
            raise Exception('Length of X and y don\'t match: {0}, {1}'.format(
                len(X), len(y)))
        Thetas_init = self.__init_thetas(self.__hl_sizes, len(X[0]), len(np.unique(y)))
        Thetas_flat = self.__flatten_arrays(Thetas_init)
        self.__n_features = len(X[0])
        self.__n_classes = len(np.unique(y)) 
        self.__labels = np.unique(y)

        # get the optimal thetas
        res = minimize(fun=self.__cost_func, 
                       x0=Thetas_flat, 
                       args=(X,y,self.__hl_sizes,self.__lam), 
                       jac=True, 
                       method=self.__solver, 
                       options={'maxiter':self.__maxiter, 
                                'gtol':self.__tol}
                       )

        self.optimal_thetas = self.__unroll_thetas(res.x, self.__hl_sizes, self.__n_classes, self.__n_features)
        return self
    
    def predict(self, X):
        """Returns predicted labels"""
        proba = self.predict_proba(X)
        pred_ix = proba.argmax(axis=1)
        pred = np.array([self.__labels[i] for i in pred_ix])
        return pred

    def predict_proba(self, X):
        """Returns probability matrix"""
        if self.optimal_thetas == None:
            raise Exception('Fit model before predicting')
        if len(X[0]) != self.__n_features:
            raise Exception('Number of features in input does not match model: {0}, {1}'.format(len(X[0]), self.__n_features))
        m = len(X)
        for t in self.optimal_thetas:
            X = np.append(np.ones((m, 1)), X, axis=1);
            t = np.transpose(t)
            X = self.__sigmoid(np.dot(X, t))
        return X
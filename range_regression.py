import numpy as np
import data
import scipy as sp
from scipy import special as spc

class RangeReg():

    def __init__(self) -> None:
        """Initialize interval linear regression model
        """
        self.x = None
        self.y = None
        self.beta = None
        self.n = None
        self.d = None
        self.x_range = None
        self.y_range = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Fit the range regression model to the data.
        Optimizing beta with constraint b >= 0

        args:
            x (np.ndarray): Interval independent variable with dimension (n x m x 2)
            y (np.ndarray): Interval dependent variable with dimension (n x 1 x 2)
        """

        self.x = x
        self.y = y
        self.n = len(self.y)
        self.d = self.x.shape[1]

        self.x_range = self.x[:,:,1] - self.x[:,:,0]
        self.y_range = self.y[:,:,1] - self.y[:,:,0]

        assert self._is_non_negative(self.x_range) and self._is_non_negative(self.y_range), "Found negative range"

        b0 = np.zeros(self.d+1)
        bounds = [(0,None) for i in range(self.d+1)]
        cons = ({'type': 'ineq', 'fun': self.b_constraint})
        res = sp.optimize.minimize(self.loss, b0, method='SLSQP', constraints=cons, bounds= bounds)
        self.beta = res.x.reshape(-1,1)

    def _is_non_negative(self, m):
        return np.min(m) >= 0
    
    def predict(self, x_intv: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x_intv (np.ndarray): Interval independent variable to be predicted with dimension (n x m x 2)

        Returns:
            np.ndarray: predicted values
        """
        # add ones in x
        x_range = x_intv[:,:,1] - x_intv[:,:,0]
        ones = np.ones(shape=(x_intv.shape[0],1))
        new_x = np.concatenate([ones,x_range], axis=1)

        y_pred = new_x @ self.beta

        return y_pred
    
    def loss(self,c):
        """ Define loss function
        """
        error_list = []

        self.beta = c.reshape(-1,1)

        # Evaluate each data point
        for i in range(self.n):
            pred_y = self.predict(np.expand_dims(self.x[i,:,:], axis=0))
            error_list.append((pred_y - self.y_range[i,:])**2)

        # sum of squared error
        l = np.sum(error_list)

        return l
    
    def b_constraint(self,b):
        return b
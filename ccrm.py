import numpy as np
import data
import scipy as sp
from scipy import special as spc
from mean_regression import MeanReg
from range_regression import RangeReg

class CCRM():

    def __init__(self) -> None:
        """Initialize interval linear regression model
        """
        self.x = None
        self.y = None
        self.beta = None
        self.n = None
        self.d = None
        self.mean_reg = None
        self.rang_reg = None
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """ Fit interval data to CCRM method:
        2 regression, mean and constrained range regression

        args:
            x (np.ndarray): Interval independent variable with dimension (n x m x 2)
            y (np.ndarray): Interval dependent variable with dimension (n x 1 x 2)
        """

        self.x = x
        self.y = y
        self.n = self.x.shape[0]

        # Fitting mean regression
        self.mean_reg = MeanReg()
        self.mean_reg.fit(self.x, self.y)

        # Fitting range regression
        self.rang_reg = RangeReg()
        self.rang_reg.fit(self.x, self.y)
    
    def predict(self, x_intv: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x_intv (np.ndarray): Interval independent variable to be predicted with dimension (n x m x 2)

        Returns:
            np.ndarray: predicted values
        """
        mean_pred = self.mean_reg.predict(x_intv)
        y_pred_center = (mean_pred[:,:,0] + mean_pred[:,:,1]) / 2

        range_pred = self.rang_reg.predict(x_intv)

        y_pred_lo = y_pred_center - 0.5*range_pred
        y_pred_up = y_pred_center + 0.5*range_pred

        y_pred = np.dstack([y_pred_lo,y_pred_up])
        return y_pred
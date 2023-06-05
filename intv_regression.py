import numpy as np
import data

class IntReg():

    def __init__(self) -> None:
        """Initialize interval linear regression model
        """
        self.x = None
        self.y = None
        self.beta = None
        self.n = None

    def fit(self,x: np.ndarray, y: np.ndarray) -> None:
        """fit the interval linear regression model to the data

        Solving equation: $\beta = (X^T X)^{-1} (X^T Y)$

        Args:
            x (np.ndarray): Interval independent variable with dimension (n x m x 2)
            y (np.ndarray): Interval dependent variable with dimension (n x 1 x 2)
        """
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        
        # add ones in x
        ones = np.ones(shape=self.y.shape)
        new_x = np.concatenate([ones,self.x], axis=1)
        self.x = new_x

        # Compute beta coefficient
        x_prod = self.intv_prod(self.x, self.x)
        xy_prod = self.intv_prod(self.x, self.y)
        x_inv = np.linalg.inv(x_prod)
        self.beta = x_inv.T @ xy_prod

    def fit_single(self,x: np.ndarray, y: np.ndarray) -> None:
        """Fit single independent variable interval linear regression

        Solving equation
        $\beta_1 = \frac{Cov(X,Y)}{Var(X)}$
        $\beta_0 = \bar{Y} - \beta_1 \bar{X}$

        NOTE: DEPRECATED, unused function

        Args:
            x (np.ndarray): Interval independent variable with dimension (n x 1 x 2)
            y (np.ndarray): Interval dependent variable with dimension (n x 1 x 2)
        """
        self.x = x
        self.y = y
        self.n = self.x.shape[0]

        cov_xy = self.intv_cov(self.x, self.y)
        var_x = self.intv_cov(self.x, self.x)

        beta_1 = cov_xy / var_x
        beta_0 = self.intv_mean(self.y) - (beta_1 * self.intv_mean(self.x))
        self.beta = np.array([beta_0, beta_1]).reshape(-1,1)
    
    def predict(self, x_intv: np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            x_intv (np.ndarray): Interval independent variable to be predicted with dimension (n x m x 2)

        Returns:
            np.ndarray: predicted values
        """
        # add ones in x
        ones = np.ones(shape=self.y.shape)
        new_x = np.concatenate([ones,x_intv], axis=1)

        y_pred_lo = new_x[:,:,0] @ self.beta
        y_pred_hi = new_x[:,:,1] @ self.beta

        y_pred = np.dstack([y_pred_lo,y_pred_hi])

        return y_pred

    def intv_mean(self, X: np.ndarray) -> float:
        """Interval empirical mean

        NOTE: DEPRECATED, unused function

        Args:
            X (np.ndarray): Input data

        Returns:
            float: Mean value
        """
        n_data = X.shape[0]
        res = X.sum()/ (2*n_data)
        return res
    
    def intv_cov(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Interval covariance

        See eq.4 on https://link.springer.com/chapter/10.1007/978-3-642-59789-3_58

        NOTE: DEPRECATED, unused function

        Args:
            v1 (np.ndarray): Interval vector 1 with dimension (n x 1 x 2)
            v2 (np.ndarray): Interval vector 2 with dimension (n x 1 x 2)

        Returns:
            float: Covariance value
        """
        term1 = 1/(4 * self.n) * np.sum([a.sum() * b.sum() for a,b in zip(v1[:,0,:], v2[:,0,:])])
        term2 = 1/(4 * self.n**2) * v1.sum() * v2.sum()
        covariance = term1 - term2

        return covariance
    
    def intv_prod(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """interval matrix product

        Product of the averaged interval matrix
        
        Args:
            x1 (np.ndarray): Interval matrix 1 with dimension (n x m x 2)
            x2 (np.ndarray): Interval matrix 2 with dimension (n x m x 2)

        Returns:
            np.ndarray: symbolic products
        """

        avg1 = (x1[:,:,0] + x1[:,:,1]) / 2
        avg2 = (x2[:,:,0] + x2[:,:,1]) / 2

        res = avg1.T @ avg2
        return res
        


if __name__ == "__main__":
    x,y = data.get_data()
    x_new = x[:,0,:]
    x = np.expand_dims(x_new, axis=1)

    if data.check_negative_interval(x) or data.check_negative_interval(y):
        raise ValueError("Negative intervals are not accepted, check your data")
    
    intreg = IntReg()
    intreg.fit(x,y)

    y_pred = intreg.predict(x)
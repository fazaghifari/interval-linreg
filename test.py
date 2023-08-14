import data
import numpy as np
from mean_regression import MeanReg
from ccrm import CCRM

x,y = data.get_data()

if data.check_negative_interval(x) or data.check_negative_interval(y):
    raise ValueError("Negative intervals are not accepted, check your data")


intreg = MeanReg()
intreg.fit(x,y)
y_pred_mean = intreg.predict(x)

ccrm = CCRM()
ccrm.fit(x,y)
y_pred_ccrm = ccrm.predict(x)

print(y_pred_ccrm)

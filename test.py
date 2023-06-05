import data
import numpy as np
from intv_regression import IntReg

x,y = data.get_data()

if data.check_negative_interval(x) or data.check_negative_interval(y):
    raise ValueError("Negative intervals are not accepted, check your data")

intreg = IntReg()
intreg.fit(x,y)

for i,b in enumerate(intreg.beta):
    print(f"Beta_{i}: {b[0]}")
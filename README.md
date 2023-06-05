# Interval Linear Regression

Attempt to re-create [(Billard and Diday, 2000)](https://link.springer.com/chapter/10.1007/978-3-642-59789-3_58) **"Regression Analysis for Interval-Valued Data"** and [(Billard and Diday, 2002)](https://www.stat.uga.edu/sites/default/files/images/Symbolic%20Data%20Analysis.pdf) **"SYMBOLIC DATA ANALYSIS: DEFINITIONS AND EXAMPLES"** -- regression part in Python

> :bulb: TL;DR : In the original paper, [(Billard and Diday, 2000)](https://link.springer.com/chapter/10.1007/978-3-642-59789-3_58) use the averaged-value of the interval to determine the regression coefficient(s).

## Linear Regression Case
Given data with 2 independent variables:

| n | $Y$ (Pulse Rate)| $X_1$ (Systolic Pressure)| $X_2$ (Diastolic Pressure)|
|---|---------|-----------|----------|
| 1 | [44-68] | [90-110]  | [50-70]  |
| 2 | [60-72] | [90-130]  | [70-90]  |
| 3 | [56-90] | [140-180] | [90-100] |
| 4 | [70-112]| [110-142] | [80-108] |
| 5 | [54-72] | [90-100]  | [50-70]  |
| 6 | [70-100]| [134-142] | [80-110] |
| 7 | [72-100]| [130-160] | [76-90]  |
| 8 | [76-98] | [110-190] | [70-110] |
| 9 | [86-96] | [138-180] | [90-110] |
| 10| [86-100]| [110-150] | [78-100] |
| 11| [63-75] | [60-110]  | [140-140]|

> Note: Row 11 is excluded because average of Diastolic $>$ Systolic

According [(Billard and Diday, 2002)](https://www.stat.uga.edu/sites/default/files/images/Symbolic%20Data%20Analysis.pdf), the interval regression model is:

$$
Y = 14.2 - 0.04 X_1 + 0.83 X_2
$$

### Running the Code

Running `test.py` yields the following result:

```
Beta_0: 14.164868347586232
Beta_1: -0.039927369272802604
Beta_2: 0.8295224023396071
```

The array shape, details of calculation is given in each function/method in the code.
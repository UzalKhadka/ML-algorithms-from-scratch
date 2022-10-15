# Linear Regression using Least Squares method

## Introduction

Linear Regression is a statistical method for finding the relationship between a dependent variable with a given set of independent variables. The relationship is modelled using a linear equation. The coefficients of the linear equation are estimated using the training data. Once the model is trained, it can be used to predict the value of the dependent variable for a given set of independent variables.
The least squares method is a statistical procedure for determining the best fit for a set of data points by minimizing the sum of offsets or residuals from the plotted curve. To predict the behavior of dependent variables, least squares regression is used. The least squares method is used to find the best fit line for a set of data points. The best fit line is the line that minimizes the sum of the squares of the vertical distances between the data points and the line.

Linear Regression Formula:
$$y = mx + c$$
where,

- y is the dependent variable
- x is the independent variable
- m is the slope of the line

For multiple independent variables, the equation becomes:
$$y = m_1x_1 + m_2x_2 + m_3x_3 + ... + m_nx_n + c$$
where,

- y is the dependent variable
- x1, x2, x3, ..., xn are the independent variables
- m1, m2, m3, ..., mn are the coefficients of the independent variables
- c is the constant

This answer in the Stack Overflow forum explains the derivation of the formula for linear regression using least squares method. <https://stackoverflow.com/questions/32114215/how-to-calculate-the-intercept-using-numpy-linalg-lstsq>

Using Least Square Method, we can solve for m and c.

The formula for m is:
$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$
where,

- $\bar{x}$ is the mean of x
- $\bar{y}$ is the mean of y

The formula for c is:
$$c = \bar{y} - m\bar{x}$$

In this notebook, we will use the Least Squares method to estimate the coefficients of the linear equation. We will use the Fish Species dataset to train the model and predict the value of the dependent variable i.e. weight of a fish for a given set of independent variables.

We can use the following performance metrics to evaluate the performance of a Linear Regression model:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-Squared Score (R2 Score)

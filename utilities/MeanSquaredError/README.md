# Mean Squared Error

## Introduction

Mean Squared Error (MSE) is a loss function that is commonly used in regression problems. It is the average of the squared differences between the predicted values and the actual values. It tells us how close the regression line is as compared to the data points. The lower the MSE, the better the model. MSE is a loss function that is differentiable, which makes it easier to optimize. MSE is also scale-invariant, meaning that it does not change if the data is scaled. This is a property that is not shared by other loss functions such as MAE.

## Use Case

We use MSE when we want to predict a continuous value. We use regression algorithms to predict continuous value unlike classification algorithms for categorical values (But we use Logistic regression, an upgraded version of traditional regression, to classify categorical values). For example, we can use MSE to predict the price of a house based on its area, number of rooms, etc. We can also use MSE to predict the number of sales of a product based on factors such as price, advertising, etc.

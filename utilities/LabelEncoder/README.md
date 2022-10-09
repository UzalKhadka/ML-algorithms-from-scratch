# Label Encoding

## Introduction

Encoding is the process of converting categorical values into numerical values. Encoding is widely used during data preprocessing because datasets may contain categorical values that cannot be used directly for training Machine Learning models. There are two major types of encoding: Label Encoding and One-Hot Encoding.
Label Encoding is one of encoding techniques that converts the unique categorical values into numerical values between 0 and n-1, where n is the number of unique categories. Label Encoding is useful when the categorical values are in a large number and the categories are ordinal. For example, the categories of a feature are “low”, “medium”, “high”, and “very high”. In this case, the categories are ordinal and the values can be encoded as 0, 1, 2, and 3. Label Encoding is also useful when the categories are not ordinal. For example, the categories of a feature are “red”, “green”, “blue”, and “yellow”. In this case, the categories are not ordinal and the values can be encoded as 0, 1, 2, and 3.

## Use Case

Here, the number of unique categories is 7. This number is too large for the One-Hot Encoding technique because it would create 7 more columns and increase the complexity of the model. Plus there is the issue of multicollinearity in the variables. Therefore, we use Label Encoding in his context.

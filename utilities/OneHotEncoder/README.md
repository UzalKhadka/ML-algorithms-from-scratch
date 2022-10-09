# One-Hot Encoding

## Introduction

Encoding is the process of converting categorical values into numerical values. Encoding is widely used during data preprocessing because datasets may contain categorical values that cannot be used directly for training Machine Learning models. There are two major types of encoding: Label Encoding and One-Hot Encoding.
One-Hot Encoding is one of the most popular encoding techniques. It is also known as Dummy Encoding. It is a process of creating new columns for each category and assigning a binary value to each column. The binary value is 1 if the category is present in the row and 0 if the category is not present in the row. The number of columns created is equal to the number of unique categories in the categorical column. The new columns are also known as dummy variables.
One-Hot Encoding is mostly used for categorical features with two or more categories. Categorical features with large number of categories are not suitable for One-Hot Encoding as they create complexity in the dimension of the dataset (i.e. Curse of Dimensionality) and it results in the complex Machine Learning model. Dummy variables also create high correlation among each other which is also called the Dummy Variable Trap. In this case, we can use Label Encoding which produces only one column for the categorical feature.

## Use Case

Here, the number of unique categories is 3. This number is perfectly fine for the One-Hot Encoding technique because it would create 3 more columns. This will not increase a lot of complexity in the dataset. Furthermore, the feature is not of ordinal type, so using Label Encoding might create some unwanted biasness on the dataset. Therefore, we use Label Encoding in his context.

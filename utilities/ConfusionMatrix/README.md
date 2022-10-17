# Confusion Matrix

## Introduction

The confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known. The confusion matrix itself is relatively simple to understand, but the related terminology can be confusing.

<pre>
                            |-----------------------------------|  
                            |             Predicted             |  
        Confusion Matrix    |-----------------------------------|  
                            |    Negative     |    Positive     |  
    |-----------------------------------------------------------|  
    |          |  Negative  |  true negative  |  false positive |  
    |  Actual  |------------|-----------------|-----------------|  
    |          |  Positive  |  false negative |  true positive  |  
    |-----------------------------------------------------------|  
    Fig: Confusion Matrix for a Binary classifier (i.e. True/False, Positive/Negative, etc.)
</pre>
<pre>

                           |----------------------------------------|  
                           |             Predicted                  |  
        Confusion Matrix   |----------------------------------------|  
                           |    Cat     |    Dog     |    Mouse     |  
    |---------------------------------------------------------------|  
    |          |    Cat    |      7     |     4      |      4       |
    |          |-----------|------------|---------------------------|  
    |  Actual  |    Dog    |      2     |     7      |      1       |
    |          |-----------|------------|---------------------------|  
    |          |   Mouse   |      1     |     0      |      9       |
    |---------------------------------------------------------------| 
    Fig: Confusion Matrix for a Multi-class classifier (i.e. Cat/Dog/Mouse)
</pre>

## Use Case

The confusion matrix is a useful tool for comparing the performance of different models. For example, we might train a logistic regression model and a random forest model, and then compare their confusion matrices to decide which model is better.

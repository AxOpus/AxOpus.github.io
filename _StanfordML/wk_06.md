---
layout: pagecollection
title: Week 6 - Evaluating a learning algorithm
collection: StanfordML
---
{% include JB/setup %}

## Debugging
Suppose you have a linear regression model for house prices, and it performs poorly on new data. What next?

- More data
- Try less features
- Try _more_ features (maybe polynomial)
- Try adjusting the regularisation parameter

These fix the problems [here](https://www.coursera.org/learn/machine-learning/supplement/llc5g/deciding-what-to-do-next-revisited).

## Evaluating a hypothesis
Often can't plot the hypothesis function (dim > 2), so we may not be able to easily visualise why our hypothesis is not behaving as expected. Usually, we split data into 70% training, and 30% test, randomly. Then we train the model, and compute the test set error, which can be square loss in linear regression, or misclassification error in logistic regression [Link](https://www.coursera.org/learn/machine-learning/supplement/aFpD3/evaluating-a-hypothesis).

## Model selection
Suppose we have 10 linear regression polynomial models, from order $$1 \dots 10$$. We fit these to the training data, and then view the test set error. Suppose that the 5th order model has the lowest error. Should we go with that model?

Not necessarily, as we have essentially 'fitted' the test set. How can we solve this? By using a validation set. We might choose to split our data into 60%/20%/20% for training, validation, and test set respectively. Now, we use our validation set to choose the model and look at the test set error.

## Bias & Variance
High bias models refer to underfitted models, where both the training and cross-validation error are high. 

High variance models refer to overfitted models, where the training error is low, but the cross-validation error is high.

#### Regularisation with Bias & Variance
Too large a regularisation parameter will introduce underfitting, whereas too small a regularisation parameter will not prevent overfitting.

To choose a regularisation parameter we might try a range of values on the training data, then select a model on validation error, and finally report that model's performance on a test set [Link](https://www.coursera.org/learn/machine-learning/supplement/JPJJj/regularization-and-bias-variance).

## Prioritisation
What is the best use of time? Collecting data, developing sophisticated features, developing algorithms to detect misspellings, etc... These are some examples of the types of elements to include in machine learning system design [Link](https://www.coursera.org/learn/machine-learning/supplement/0uu7a/prioritizing-what-to-work-on).

Recommended approach is to start with a simple model, and try and graph relevant plots. Errors might be analysed manually to see if there are any traits which seem to be common. This may be used to determine what avenues to explore in more detail, for example which features to add. This is known as error analysis [Link](https://www.coursera.org/learn/machine-learning/supplement/Z11RP/error-analysis).

## Skewed classes
Suppose you build a logistic regression classifier which obtains 99% accuracy on a test set. This appears good, however suppose that the rate of cancer is 0.5%. Then a classifier which simply outputs "no cancer" for each input will have a lower error rate than out logistic one. 

This is an example of skewed classes where we have significantly more examples from one class than another. To combat this, we can use precision and recall and the F-score instead [Link](https://www.coursera.org/learn/machine-learning/lecture/CuONQ/trading-off-precision-and-recall).

## Data
More data = better. This isn't the entire picture, because we may have a huge amount of data about the size of houses, but that alone is not enough to predict the house's worth. A sensible measure is if the data contains enough features for a human to make a prediction, then more of that data will generally improve the algorithm's performance.









---
layout: pagecollection
title: Week 3 - Logistic Regression
collection: StanfordML
---
{% include JB/setup %}

## Logistic Regression - Classification
Classification examples:
- Spam/not-spam
- Fraudulent transaction (y/n)
- Tumour (Benign/Non-benign)

Predicting $$y \in \{0, 1\}$$, for a binary classifier. This set can be extended to multi-classification problems e.g. $$y \in \{0, 1, \dots n\}$$.

If we are classifying binary data, then $$y \in \{0, 1\}$$, but our linear regression function is not constrained to lie in this range. We could threshold it, such that $$h_{\theta}(x) < 0.5$$ is class 0, and $$h(\theta) \geq 0.5$$ is class 1.

Logistic regression outputs values $$ 0 \leq h\_{\theta}(x) 1 $$ and we define $$ h\_{\theta}(x) = g(\theta^T x) = \frac{1}{1 + exp(-\theta^T x)}$$ [Link](https://www.coursera.org/learn/machine-learning/supplement/AqSH6/hypothesis-representation).

#### Decision boundary
Where $$\theta^T x \geq 0$$ we define $$y = 1$$ [Link](https://www.coursera.org/learn/machine-learning/supplement/N8qsm/decision-boundary).

## Cost function
The cost function isn't the same as in linear regression, because this would give us non-convexity. Instead it is a log function which $$\rightarrow 0$$ as we approach the correct class, and $$\rightarrow \infty$$ as we approach the incorrect class [Link](https://www.coursera.org/learn/machine-learning/supplement/bgEt4/cost-function).

#### Gradient descent
We can simplify the cost function into a single line, then use gradient descent. The update rule is identical to linear regression, but works because we have changed the hypothesis function [Link](https://www.coursera.org/learn/machine-learning/supplement/0hpMl/simplified-cost-function-and-gradient-descent).

#### Other optimisation algorithms
Gradient descent is one algorithm for optimisation but others exist. The ones mentioned are: Conjugate gradient, BFGS, and L-BFGS [Link](https://www.coursera.org/learn/machine-learning/supplement/cmjIc/advanced-optimization).

## Logistic Regression - Multi-Class Classification
Consider foldering/tagging: Work, Friends, Family, Hobby... etc. Define these as $$y \in [1, 2, 3, \dots n]$$.

We can use a technique called: One vs. All (One vs. rest). This turns the problem into a repeated binary classification problem, where all the training examples not in the target class are treated as $$y = 0$$. We train a separate classifier for each class [Link](https://www.coursera.org/learn/machine-learning/supplement/HuE6M/multiclass-classification-one-vs-all).

## Overfitting
We may have very low training error, but the model is not generalisable. This may be due to "overfitting" the training data. For example, with 5 points we can choose a fifth-order polynomial which can fit the data directly, but will be very "wiggly" and unlikely to make good predictions [Link](https://www.coursera.org/learn/machine-learning/supplement/VTe37/the-problem-of-overfitting).

#### Addressing overfitting
We can reduce the number of features either manually, or model selection. However, we more usually use regularisation.

In regularisation we add a term to penalise large weights. This means that when we optimise, we will look to keep the weights small [Link 1](https://www.coursera.org/learn/machine-learning/supplement/1tJlY/cost-function), [Link 2](https://www.coursera.org/learn/machine-learning/supplement/pKAsc/regularized-linear-regression), [Link 3](https://www.coursera.org/learn/machine-learning/supplement/v51eg/regularized-logistic-regression).


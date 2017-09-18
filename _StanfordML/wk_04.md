---
layout: pagecollection
title: Week 4 - Neural Networks
collection: StanfordML
---
{% include JB/setup %}

## Non-linear hypothesis
Given a logistic regression problem, with a decision boundary that is likely to be complex we can model this by adding polynomial terms. With two base features this may be possible, but as we add features the function ends up with a huge number of terms: if we just add quadratic terms, $$\approx \mathcal{O}(n^2)$$; for cubic terms,  $$\approx \mathcal{O}(n^3)$$.

## Neural representation
Inputs: $$x_0, x_1, \dots, x_n$$ feed in to a 'unit' which has a function, for example a logistic unit (also called a sigmoid activation function), which then outputs some value.

Inputs are arranged in layers, which can be stacked. The first layer is called the input layer; the last layer, the output layer; and any intermediate layers, hidden layers.

Each layer has a weight matrix associated with it. If we denote the number of units in layer $$j$$ as $$s_j$$, and the number of units in layer $$j+1$$ as $$s_{j+1}$$, then the weight matrix is of size: $$(s_{j+1}, s_{j}+1)$$ [Link](https://www.coursera.org/learn/machine-learning/supplement/Bln5m/model-representation-i).

## Forward-propagation
We can send information forward in a vectorised manner [Link](https://www.coursera.org/learn/machine-learning/supplement/YlEVx/model-representation-ii)

## Representation
A classic use of why a non-linear function is useful is the **AND** function [Link](https://www.coursera.org/learn/machine-learning/supplement/kivO9/examples-and-intuitions-i). We can also represent other boolean combinations, which we can combine to obtain more complex functions [Link](https://www.coursera.org/learn/machine-learning/supplement/5iqtV/examples-and-intuitions-ii).

## Multiclass classification
For $$n$$ classes we use an output vector of size $$n$$, where we want to represent the identified class as 1, and the other classes as 0 [Link](https://www.coursera.org/learn/machine-learning/supplement/xSUml/multiclass-classification).



---
layout: pagecollection
title: Week 1 - Machine Learning Overview
collection: StanfordML
---
{% include JB/setup %}

## Overview
Intoduction to ML. Overview of the course:

- Supervised learning:
    - Supply labelled data, from which to make predictions.
    - i.e. Given ''right answer'' for an input.
    - Regression, predicting real numbers.
    - Classification, predicting which group.
- Unsupervised learning:
    - Unlabelled data.
    - Trying to find patterns in this data automatically.

## Cost function
- Want to minimise some function $$J(\theta_{0}, ..., \theta_{n})$$.
- Least squares $$\sum(\hat{y} - y)^2$$.
- If $$J(\theta_0)$$, can plot as a 2D plot; if $$J(\theta_0, \theta_1)$$, can plot as a contour.
- An algorithm to minimise this cost function is gradient descent.
    - Pick a point 
    - Take a step in the direction of steepest descent
        - Step size is set manually, or by optimisation algorithm (e.g. Adam)
        - Termed the "learning rate"
        - Too small, will take a long time; too large, may diverge
    - Repeat until reached a **local** minima
    - All new theta\_i values are updated simultaneously
    - "Batch", uses all training examples; 
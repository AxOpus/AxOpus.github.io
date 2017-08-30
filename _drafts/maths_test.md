---
layout: post
category : lessons
tags : [machine learning, AI, automation]
---
{% include JB/setup %}

## Overview
This is a curated list of Machine learning (ML) and Artificial Intelligence (AI) resources.

### Contents

## Starter materials


- Supervised learning:
    - Supply labelled data, from which to make predictions.
    - i.e. Given ''right answer'' for an input.
    - Regression, predicting real numbers.
    - Classification, predicting which group.
- Unsupervised learning:
    - Unlabelled data.
    - Trying to find patterns in this data automatically.

## Cost function
- Want to minimise some function J(theta\_0, ..., theta\_n).
- Least squares sum(y\_hat - y)^2.
- If J(theta\_0), can plot as a 2D plot; if J(theta\_0, theta\_1), can plot as a contour.
- An algorithm to minimise this cost function is gradient descent.
    - Pick a point 
    - Take a step in the direction of steepest descent
        - Step size is set manually, or by optimisation algorithm (e.g. Adam)
        - Termed the "learning rate"
        - Too small, will take a long time; too large, may diverge
    - Repeat until reached a **local** minima
    - All new theta\_i values are updated simultaneously
    - "Batch", uses all training examples; 
- This is some maths: $$\alpha = x^2$$






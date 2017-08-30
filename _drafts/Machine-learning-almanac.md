---
layout: post
category : lessons
tags : [machine learning, AI, automation]
---
{% include JB/setup %}

## Overview
This is a curated list of Machine learning (ML) and Artificial Intelligence (AI) resources. There is a huge amount of information available, but this is collection of some books, courses, blogs to start. 

#### Contents

## Starter materials
**[Andrew Ng's Coursera Course](https://www.coursera.org/learn/machine-learning/home/welcome) (2011)** A solid foundational course in ML. Covers supervised and unsupervised learning in a fairly simple manner, so you don't get too bogged down in the maths. Although the field has advanced significantly since, particularly in deep learning, the principles remain. It uses Octave (or Matlab), which is useful if you only know python.

**[CS229-MachineLearning](https://see.stanford.edu/Course/CS229/) (2008)** The course which Andrew Ng teaches at Stanford. Covers similar material to the above course, but with more maths. These course materials and lectures are from 2008, but you can find the most recent notes [here](http://cs229.stanford.edu).

**[Udacity - Introduction to Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120) (2016)** Taught by [Sebastian Thrun](https://en.wikipedia.org/wiki/Sebastian_Thrun) and [Katie Malone](http://blog.udacity.com/2016/04/women-in-machine-learning-katie-malone.html), this covers some slightly different material than the Andrew Ng course, with more focus on data science.

## Background Programming

## Background Mathematics

#### Linear Algebra

#### Probability

#### Calculus

## Books

## Deep Learning

## Reinforcement Learning

## Communities

## 





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
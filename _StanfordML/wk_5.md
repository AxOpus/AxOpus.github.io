---
layout: pagecollection
title: Week 5 - Neural Networks Learning
collection: StanfordML
---
{% include JB/setup %}

## Cost function
Focus on NNs for classification. We will generalise the logistic regression loss cost function in our NNs [Link](https://www.coursera.org/learn/machine-learning/supplement/afqGa/cost-function).

## Backpropagation
We have previously used forward propagation to calculate the activation functions in each layer. This will result in us calculating some outputs. We then work out the difference between our outputs and the actual values, and "backpropagate" these error signals through the network [Link](https://www.coursera.org/learn/machine-learning/supplement/afqGa/cost-function).

_Note: For MATLAB optimisation algorithms like ```fminunc``` we need to unroll our weight (or parameter) matrices into vectors [Link](https://www.coursera.org/learn/machine-learning/supplement/v88ik/implementation-note-unrolling-parameters) ._

#### Gradient checking
We can numerical check our gradients are computed correctly. If we have some gradient $$J'(\theta)$$ we can compute: $$\frac{J(\theta + \epsilon) - J(\theta - \epsilon)}{2 * \epsilon}$$ where $$\epsilon \approx 10^{-4}$$ and we should find this approximates the gradient.

#### Initialisation
The choice of initial weights is very important. Initialising to zero has undesirable properties, namely it creates symmetry which means the weights are the same going to each activation function from the unit in the previous layer. To get around this, we initialise the weights randomly, typically with Xavier initialisation [Link](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization), or some other random initialisation [Link](https://www.coursera.org/learn/machine-learning/supplement/KMzY7/random-initialization).






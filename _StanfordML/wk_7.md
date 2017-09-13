---
layout: pagecollection
title: Week 7 - Large margin classification
collection: StanfordML
---
{% include JB/setup %}

## Support Vector Machine

#### Optimisation objective
We can split the cost function in logistic regression into two components: the $$-y$$ and $$(1-y)$$ parts. We assign two new cost functions which are a straight line approximation to the logistic regression functions shown below:

<figure class="figure">
  <img src="http://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[12].png" class="img-responsive center-block"  alt="SVM graphs">
  <figcaption class="figure-caption text-center">Image 1. SVM graphs</figcaption>
</figure>

We now get rid of the $$\frac{1}{m}$$ terms, since this is an optimisation problem, and express the resulting function as $$\mathbf{A} + \lambda \mathbf{B}$$. The standard notation for this is $$\mathbf{C}\mathbf{A} + \mathbf{B}$$, where if $$\mathbf{C} = \frac{1}{\lambda}$$, then the two expressions give the same value.

A large value of $$\mathbf{C}$$ means the SVM tries to fit all examples correctly.

#### Large margin
Suppose $$\mathbf{C}$$ is very large, then since we are optimising $$\mathbf{C}\mathbf{A} + \mathbf{B}$$, we want $$\mathbf{A}$$ to be $$0$$. This occurs when $$\theta^{T} \mathbf{x} \geq 1$$.

The consequence of this is that SVM doesn't want to get the classification _just_ right, it wants to include a safety margin as well, as shown by the blue lines below:

<figure class="figure">
  <img src="http://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[15].png" class="img-responsive center-block"  alt="SVM graphs">
  <figcaption class="figure-caption text-center">Image 2. SVM margins</figcaption>
</figure>

## Kernels
Suppose we want to find a non-linear decision boundary. We could use high-order polynomials, but we saw that those might be computationally expensive. Instead, we can use other functions for example Gaussian kernels: $$f_i = exp(-\frac{||x - l^{(i)}||^2}{2 \sigma^2})$$. We define $$l$$ as a landmark, which has its centre on one of the training examples.

We encode all of these functions in a vector, $$\mathbf{f}$$, such that we calculate $$\mathbf{\theta}^T \mathbf{f}$$. This will then produce a value, which we use to create the decision boundary.

<figure class="figure">
  <img src="http://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[49].png" class="img-responsive center-block"  alt="SVM graphs">
  <figcaption class="figure-caption text-center">Image 3. Kernel classification example</figcaption>
</figure>

<figure class="figure">
  <img src="http://www.holehouse.org/mlclass/12_Support_Vector_Machines_files/Image%20[50].png" class="img-responsive center-block"  alt="SVM graphs">
  <figcaption class="figure-caption text-center">Image 4. Kernel boundary example</figcaption>
</figure>
















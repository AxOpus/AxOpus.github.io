---
layout: pagecollection
title: Week 8 - Unsupervised Learning
collection: StanfordML
---
{% include JB/setup %}

## Clustering
In supervised learning we have input pairs: $$\{(x^1, y^1), (x^2, y^2), \dots, (x^n, y^n)\}$$, which we can use to predict values or classifications for unseen inputs. In unsupervised learning, we just have inputs: $$\{x^1, x^2, \dots, x^n\}$$, and we would like to find structure in the data.

Suppose we graph some data, as shown below:

<figure class="figure">
  <img src="http://www.holehouse.org/mlclass/13_Clustering_files/Image.png" class="img-responsive center-block"  alt="SVM graphs">
  <figcaption class="figure-caption text-center">Image 1. Unlabelled data</figcaption>
</figure>

We may see that there are two 'clusters' of points. An unsupervised algorithm which can find these clusters is called a clustering algorithm.

#### K-means
K-means is a clustering algorithm. It works by defining "clustering centroids" which will iteratively move into the euclidean central mean of each cluster. In pseudocode:

```python
Randomly initialise K cluster centroids: mu_1, mu_2, ... mu_K
Repeat until converged{
    for i = 1:m
        c_i = index of cluster centroid closest to x_i
    for = k = 1:k
        mu_k = mean of points assigned to cluster k
}

```

The random initialisation is done by selecting k data points at random and running the algorithm.

The number of clusters is often chosen manually

#### Optimisation objective
The optimisation objective is simply minimising the distance between a point and its assigned cluster centroid i.e. $$J(c^1, c^2, \dots, c^n) = \frac{1}{m}\sum_{i=1}^{m}|| x^i - \mu_{c^i}||^2$$.

## Dimensionality reduction
### Data compression

Sometimes we may have redundant features (e.g. a feature which is length in cm and a feature which is length in inches). If we project these features onto a single line, we have reduced the dimensionality of the problem and reduced the amount of data we need to store.

#### Principal components analysis (PCA)

PCA is a technique for reducing the dimensionality of a problem. Suppose we have data in 2D, PCA would find a 1D line to which the points are mapped. It finds this line by minimising the "projection error", which is the squared distance between the each point and the line.

Note, PCA is not linear regression:

<figure class="figure">
  <img src="https://wikidocs.net/images/page/4870/dim202.PNG" class="img-responsive center-block"  alt="SVM graphs">
  <figcaption class="figure-caption text-center">Image 1. PCA vs. Linear Regression</figcaption>
</figure>

We perform Singular Value Decomposition (SVD) on the data, to obtain our eigenvector matrix $$U$$. From this, we select the first $$k$$ columns, corresponding to the number of dimensions we wish to reduce the data to. We then perform $$U^T \mathbf{x}$$ to obtain our reduced data.

We choose the "best" number of dimensions, by starting with $$k=1$$ and measure how much of the variance is retained. We repeat this process, incrementing $$k$$ by 1, until we have retained an appropriate (90 - 99%) amount of the variance.




















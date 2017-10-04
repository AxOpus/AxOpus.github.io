---
layout: post
category : lessons
tagline: (and you)
tags : [machine learning, AI, automation]
---
{% include JB/setup %}

# Overview

A significant amount of success has been demonstrated recently in the field of computer vision. A large portion of this success has been due to the implementation of neural networks, and specifically deep Convolutional Neural Networks (CNNs).

This network architecture is everywhere: every photo uploaded to FaceBook is analysed by three separate CNNs; by [Microsoft](https://blogs.microsoft.com/firehose/2017/08/21/microsoft-researchers-achieve-new-milestone-in-conversational-speech-recognition/) in conversational speech recognition; and by [Google](https://googlesystem.blogspot.co.uk/2013/06/how-googles-image-recognition-works.html#gsc.tab=0) for photo search.

In these articles, we'll cover the details of this architecture in (hopefully!) an easy to follow manner. This is structured as follows:

Part 1:
- [1. Background](#1-background)
  - [1.1 Motivation](#11-motivation)
  - [1.2 Images](#12-images)
- [2. Components](#2-components)
  - [2.1 Kernel](#21-kernel)
  - [2.2 Strides](#22-strides)
  - [2.3 Padding](#23-padding)
  - [2.4 Activation and pooling](#24-activation-and-pooling)
  - [2.5 Putting it all together](#25-putting-it-all-together)

Part 2:
- [3. Implementation](#3-implementation)
  - [3.1 Forward propagation](#31-forward-propagation)
    - [3.1.1 Maths](#311-maths)
    - [3.1.2 Python](#312-python)
  - [3.2 Backward propagation](#32-backward-propagation)
    - [3.2.1 Maths](#321-maths)
    - [3.2.2 Python](#322-python)
  - [3.3 TensorFlow](#33-tensorflow)
  - [3.4 PyTorch](#34-pytorch)
- [4. Example](#4-example)

We presume you have some familiarity with neural networks (at least to the level of understanding the forward and backward passes) and Python. Additionally we will also reference supporting material where appropriate.

# 1. Background
### 1.1 Motivation
CNNs take advantage of data which is spatially related. For example in an image, or in a time history. The principle idea is that by looking at smaller local regions of the data, we might find something that looks like an edge, or an eye. We can then combine these to try and determine what is in the image.

In traditional feedforward NNs, we would be creating large weight matrices linking each pixel to each hidden neuron. This can create models with very large numbers of parameters, and treat pixels at opposite sides of the image the same as pixels close together.

These models can be difficult (or impossible) to train, and furthermore they don't seem to have much grounding in the visual domain. In contrast, CNNs have some biological arguments supporting their structure. More information can be found in **[Deep Learning - 9.10](http://www.deeplearningbook.org/contents/convnets.html)** if you want to read more.

### 1.2 Images
We have made reference to pixels in the above paragraphs, and this is key to understanding how a computer 'sees' images. At a simple level, these pixels are combinations of three miniscule LEDs which output red, green, or blue light, commonly referred to as [RGB](http://rgb.to). The intensity of each light varies from 0-255, where 0 is totally off, and 255 is totally on. So, we can represent a black pixel by: 0, 0, 0; a white pixel by: 255, 255, 255; and a very red pixel by 255, 0, 0.

This allows us to represent an image as a 3D matrix where the height and width of the image are 2nd and 3rd dimensions, and the RGB values are represented by the 1st, depth, dimension. It is standard convention in CNNs to refer to this representation as: ```C x H x W```, where ```C``` is the 'channel' of the RGB value.  

As an aside, you may feel that this representation severely limits computer vision, after all our eyes don't _see_ in pixels, do they? Well, that is true, however consider the following (inspired by [this](https://mathoverflow.net/questions/25983/intuitive-crutches-for-higher-dimensional-thinking) discussion.). Here is an image which is $$640 \times 480$$ pixels (width $$\times$$ height):

<figure class="figure">
  <img src="http://i.ytimg.com/vi/uAu3jQWaN6E/sddefault.jpg" class="center-block img-responsive" alt="Geoffrey Hinton">
  <figcaption class="figure-caption text-center">Fig 1. Geoffrey Hinton rendered at 640 x 480</figcaption>
</figure>

Now, consider this image as being a stacked series of vectors in $$\mathbb{R}^{640}$$. We could move _through_ this space, adjusting values one at a time as appropriate until we reached a combination that looked like this:

<figure class="figure">
  <img src="https://i.ytimg.com/vi/iyLqU6MpfO0/sddefault.jpg" class="center-block img-responsive" alt="Yoshua Bengio">
  <figcaption class="figure-caption text-center">Fig 2. Yoshua Bengio rendered at 640 x 480</figcaption>
</figure>

Now imagine all the images that we would have seen on that journey from one image to another. They all live in the same vector space, and each individual image occupies one part of that space. It is possible, therefore, to produce a $$640 \times 480$$ image which is a representation of _anyone who has ever lived or ever will live_. That's right. By adjusting those pixel values we could obtain representational images of Caeser, Aristotle, [George Eastman](https://en.wikipedia.org/wiki/George_Eastman), or anyone who will ever be born. That is really incredible.

You will note that I have been using the term 'representation'. We could increase or decrease the size of the image, and we would still be able to obtain some reasonable depiction of the individual, down to some image size at least. So, for example, a photo of 'you' would still be 'you' whether it is at $$640 \times 480$$, $$1024 \times 768$$, or $$4096 \times 3072$$.


# 2. Components

Now that we have an understanding of what a neural net is for, and how images are represented, we can move to a deeper level of understanding CNNs.

### 2.1 Kernel
The kernel is really the core concept behind in a CNN. If you recall from above, the CNN is interested in local regions, and determining if there is, for example, an edge, or an eye. These are called 'features', which is why kernels in a CNN are also known as 'feature extractors', or 'filters'. These terms are used interchangeably.

Imagine you want to blur an image. One way of doing so would be to take the top left pixel in your image, look at the pixels immediately surrounding it, take the mean value of these and add this pixel to a new image. Then you could move one pixel right, and do the same process. You could repeat this for the entire image until you have created a new image which is a blurred representation of your original.

So, for pixel $$\mathbf{e}$$, below, we would do the following:

$$
\mathbf{A}=
\left[
 \begin{matrix}
  a & b & c \\
  d & \mathbf{e} & f \\
  g & h & i
 \end{matrix}
\right]
\odot
\frac{1}{9}
\left[
 \begin{matrix}
  1 & 1 & 1 \\
  1 & 1 & 1 \\
  1 & 1 & 1
 \end{matrix}
\right]
$$

$$
\text{blurred } \mathbf{e} = \sum_{m=1}^{3}\sum_{n=1}^{3}\mathbf{A}
$$

Where $$\odot$$ is the [hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) of matrices.

This can be easily represented in code:

~~~python
# Let E be a 3x3 matrix from 'a' to 'i'
blurred_e = np.sum(E * np.ones((3, 3) * 1.0/9))
~~~

What we've done here is compute the convolution between these two matrices (strictly speaking the cross-correlation) to produce an output value. Congratulations, you've just taken a first step into understanding CNNs. If you look at **Example 1.1** in the Jupyter notebook, you will see this working on a test image. A further treatment on this idea is found [here](https://www.cs.cornell.edu/courses/cs1114/2013sp/sections/S06_convolution.pdf).

Now, each combination of matrix entries in our ```3 x 3``` kernel will produce some different output when convolved with our input image. We can see this in **Example 1.2** in the Jupyter notebook. If we convolve these kernels with the images we can produce an output which is made up of the edges of the image. A useful explanation of this can be found in slides 51 onwards of [this](https://www.cs.toronto.edu/~urtasun/courses/CV/lecture02.pdf) presentation.

You can play around with adjusting the kernel values in the notebook, or you can use [this](http://setosa.io/ev/image-kernels/) excellent website.

So this is all well and good, but this is still quite abstract. Let's go through a concrete example. If you look at **Example 1.3** in the jupyter notebook you will see two kernels which are the cropped eyes of the image. We go across all the pixels in the image convolving the left eye kernel at each point. It can be seen that when the kernel passes over the left eye the output image at this point is extremely bright. This means that a left eye has been found in an image. A similar behaviour can be seen with the right eye kernel.

Imagine now we have lots and lots of these kernels, each taking on a separate feature. For example, imagine separate kernels for a mouth, nose, left, and right eye of a person. Then, if we were to convolve each of this with an image, and all these features were detected, then it is likely that we are looking at an image of a person.

The most important thing to remember is that the CNN is trying to find the values (weight) for these kernels. It is these that are the parameters of the network, and the network will learn these features.

### 2.2 Strides
So far we have been 'sliding' our kernel across the image one pixel at a time. This produces an output image which is very close to the same size as our input. This is usually beneficial, but on occasion we may be able to move 2 or even 3 pixels at a time, without a significant loss in performance.

This length is called a 'stride'. For most applications we will use a stride of 1, because, as we will see, there are other ways of controlling the size of the output. A couple of images showing different strides are shown at the start of [this](https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/) article.

### 2.3 Padding

In the examples we have been using in the jupyter notebook, you may have noticed that there is something called 'padding'. This is sometimes necessary in convolution, because we need to preserve dimensionality. For example, if we recall our blur example from above, we convolved a ```3 x 3``` matrix with an image, starting in the top left corner of the image.

This means that part of our filter is outside the image, so to combat this we 'pad' the image with values, almost always zeros. If we didn't include this padding, then, in this example, we would be constrained to sliding the kernel along an image which is 2px smaller in both height and width. Can you see why?

Padding is also used to control the size of the output image. Given an input image of size $$w1 \times h1$$ then the output size is defined as:

$$
s = \text{stride length}\\
p = \text{padding}\\
k = \text{kernel size}\\
w2 = \frac{w1 - k + 2 \times p}{s} + 1\\
h2 = \frac{h1 - k + 2 \times p}{s} + 1\\
$$

We will usually be dealin with square images such that $$w1 = h1$$. Often we will want to maintain the image size after convolution, that is $$w2 = w1$$, and have a stride of 1. It can be useful to remember that the amount of padding we need, in this specific case, is:

$$
p = \frac{k-1}{2}
$$

Note that this can lead to combinations of padding, kernel size, image, and stride which do not divide into integers. Most CNN implementations will have some way of handling this, usually by increasing padding. However it is advisable to try and keep the combinations divisible, at least when starting out with CNNs, to aid with debugging.

### 2.4 Activation and pooling

After we have performed the convolution operation we obtain the output image. As you will be familiar with if you have worked with neural networks, we then apply an elementwise activation function, such as ReLU or ELU. This introduces non-linearity into the function, so that our layers aren't simply linear combinations.

To control the number of parameters it is common to insert a pooling layer after the output of the activation layer. Typically this is max-pooling, in which the maximum of a small grid of the input image is inserted into the pooling layer.

Typically, strides of 2 are used with a grid size of ```2 x 2``` such that the output of the pooling layer is half the width and height of the input. This can be seen in the following graphic:

<figure class="figure">
  <img src="https://upload.wikimedia.org/wikipedia/commons/e/e9/Max_pooling.png" class="center-block img-responsive" alt="Max pooling">
  <figcaption class="figure-caption text-center">Fig 3. Max pooling example</figcaption>
</figure>

### 2.5 Putting it all together
So far we have seen the components of a CNN in isolation. The most basic CNN architecture may look like:

```python
Input image -> Convolution -> ReLU -> Max Pool -> FC -> ReLU -> FC -> SoftMax
```

We recall that the images we are dealing with are of shape: ```C x H x W```. Although we have shown the kernel convolving on a 2D image, it is important to note that the kernels need to be of the same depth as the image. So our kernels will be of size: ```C x k x k```. An excellent example of this is shown [here](http://cs231n.github.io/assets/conv-demo/index.html).

Now, we said that there can be multiple kernels, so we can consider a single kernel to be of size: ```1 x C x k x k```, which means that ```K``` kernels will be of size: ```K x C x k x k```. This may be slightly complicated to try and visualise, so it may become clearer in Part 2, when we show the implementation of it.

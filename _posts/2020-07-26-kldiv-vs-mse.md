---
layout: post
mathjax: true
title: On the relation of learning Gaussian and the L2 loss

# optional stuff
tags: [machine learning, gaussians, mse, loss]
feature-img: "assets/img/introfigure_kldiv_l2loss.png"
---

In [my previous post]({% post_url 2020-07-12-gaussian-part1 %}){:target="_blank"}, I introduced Gaussians and their existence in different datasets. This post is primarily aimed at a derivation that I think is trivial for understanding where Gaussians are used in machine learning. This proof can be found in any machine learning textbook. Most part of the derivation can be found in this [StatsExchange post](https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians){:target="_blank"} that I got help from. 

We start with defining the widely used Kullback-Leibler Divergence metric that measures the difference between two distributions. Next we will look at how it may be applied to a typical machine learning model. Finally, we will go through the derivation that proves its relation to the L2 loss that is widely used for learning models that address regression tasks.

## Kullback-Leibler Divergence
[Kullback-Leibler Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) measures how different a given distribution is from a reference distribution. For two distributions, $$P(x)$$ and $$Q(x)$$, it is defined as:

$$ D_{KL}(P \parallel Q) = \int_x P(x) \log \big(\frac{P(x)}{Q(x)}\big) \,dx, $$

In context of machine learning, $$P(x)$$ can be thought of the underlying groundtruth distribution that we are trying to learn using our model distribution $$Q(x)$$. As I will explore through derivations in the following section, this measure is much more common in machine learning as most regression methods are specific case of the above equation.

> KL divergence is called divergence and not a distance measure, as it really does not measure distance between two distributions. For any measure to qualify as distance measure, it needs to be symmetric, which is not the case for KL divergence, i.e. for two distributions $$P$$ and $$Q$$, $$D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)$$. You can also understand symmetry property of distance measures by looking at distance between two points, e.g. Euclidean distance between point $$A$$ and $$B$$ would be the same no matter which of the two is our starting point ($$d(A, B) = d(B, A)$$).

## KL Divergence in data-driven machine learning
In order to understand how a data-driven machine learning (ML) method utilises KL divergence, let's first start with a simple example. Let a groundtruth distribution $$P_{gt}(x)$$ be defined as a one-dimensional Gaussian:

$$ P_{gt}(x) = \frac{1}{\sigma_{gt} \sqrt{2\pi}} e ^ {-\frac{1}{2} \big[\frac{(x-\mu_{gt})^2}{\sigma_{gt}^{2}}\big]}, $$

where $$\mu_{gt}$$ is the mean and $$\sigma_{gt}$$ is the standard deviation of the distribution. It may become clear later, however, we can think of $$P_{gt}(x)$$ in terms of the distribution of a dataset which an ML method may learn.

Now let's think of a black box machine learning model that tries to model $$P_{gt}(x)$$ by inferring the model distribution $$P_{m}(x)$$ as shown in figure below.

{% include image.html src="/assets/blog/kldiv-gaussians-part2a/diagrams-SimpleMLModelPm.png" alt="ml model with output P_m" caption="" width=400 %}

Our black box ML model can be any method, e.g. a Linear Regression model. I have intentionally left it a black box to focus on the KL divergence aspect of learning.

What could $$P_m(x)$$ be modelled as? If you guessed a Gaussian, you are totally right! $$P_m(x)$$ is defined as:

$$ P_{m}(x) = \frac{1}{\sigma_{m} \sqrt{2\pi}} e ^ {-\frac{1}{2} \big[\frac{(x-\mu_{m})^2}{\sigma_{m}^{2}}\big]}, $$

where $$\mu_{m}$$ is the mean and $$\sigma_{m}$$ is the standard deviation of the distribution.

> **For brevity I omit the function arguments $$(x)$$ from the derivations that follow.**

We now need to define a measure that can guide our black box ML model to be able guess a $$P_m$$ that is as close to $$P_{gt}$$ as possible. This can be defined using the KL divergence between the two distributions as follows:

$$ D_{KL}(P_{gt} \parallel P_{m}) = \int P_{gt} \log \big(\frac{P_{gt}}{P_{m}}\big) \,dx $$

KL divergence measures how far a predicted distribution $$P_m$$ is from the original distribution $$P_{gt}$$. The problem of learning the black box ML model can then be posed as minimisation problem:


$$ \underset{\mu_m, \sigma_m}{arg\,min} \, D_{KL}(P_{gt} \parallel P_{m}) = \underset{\mu_m, \sigma_m}{arg\,min} \, \int P_{gt} \log \big(\frac{P_{gt}}{P_{m}}\big) \,dx $$

## Relation of KL Divergence to L2 loss
Now let's start with the actual derivation that proves learning with L2 loss to be a special case of learning with KL divergence between two distributions. We start with the equation and expand the logarithm terms:

$$ D_{KL}(P_{gt} \parallel P_{m}) = \underbrace{\int P_{gt} \log \big(P_{gt}\big) \,dx}_{T1} - \underbrace{\int P_{gt} \log \big(P_{m}\big) \,dx}_{T2} $$

### Focusing on term $$T1$$
Looking at the first term $$T1$$ in the equation, we can start simplifying it further:

$$ \int P_{gt} \log \big(P_{gt}\big) \,dx = \int P_{gt} \log \big( \frac{1}{\sigma_{gt} \sqrt{2\pi}} e ^ {-\frac{1}{2} \big[\frac{(x-\mu_{gt})^2}{\sigma_{gt}^{2}}\big]}\big) \,dx $$

We can expand the logarithm terms:

$$  = \int P_{gt} \log \big( \frac{1}{\sigma_{gt} \sqrt{2\pi}} \big) \,dx + \int P_{gt} \log \big( e ^ {-\frac{1}{2} \big[\frac{(x-\mu_{gt})^2}{\sigma_{gt}^{2}}\big]}\big) \,dx $$

$$  = \log \big( \frac{1}{\sigma_{gt} \sqrt{2\pi}}  \big) \int P_{gt} \,dx - \int \frac{1}{2} \big[\frac{(x-\mu_{gt})^2}{\sigma_{gt}^{2}}\big] P_{gt} \log \big( e \big) \,dx $$

We know that $$\log{e}=1$$. Moreover, $$P_{gt}$$ is a Gaussian distribution, therefore $$\int P_{gt} \,dx=1$$.

Now lets expand the quadratic term:

$$  = -\log \big( \sigma_{gt} \sqrt{2\pi}  \big) -\frac{1}{2\sigma_{gt}^{2}} \int  (x-\mu_{gt})^2 P_{gt} \,dx $$

$$  = -\log \big( \sigma_{gt} \sqrt{2\pi}  \big) -\frac{1}{2\sigma_{gt}^{2}} \big[\int x^2 P_{gt} \,dx - 2 \mu_{gt} \int x P_{gt} \,dx + \mu_{gt}^2 \int P_{gt} \,dx  \big] $$

$$\int x P_{gt} \,dx$$ is the expected value of $$x$$ for distribution $$P_{gt}$$, i.e.$$\int x P_{gt} \,dx = \mathop{\mathbb{E}}_{P_{gt}}[x]$$. As $$P_{gt}$$ is a Gaussian distribution, we know that its expected value is $$\mathop{\mathbb{E}}_{P_{gt}}[x] = \mu_{gt} $$. We can further simplify our equation as:

$$  = -\log \big( \sigma_{gt} \sqrt{2\pi}  \big) -\frac{1}{2\sigma_{gt}^{2}} \big[\int x^2 P_{gt} \,dx - 2 \mu_{gt}^2  + \mu_{gt}^2 \big] $$

From [this answer](https://math.stackexchange.com/questions/99025/what-is-the-expectation-of-x2-where-x-is-distributed-normally){:target="_blank"}, we know the following relationship:

$$ \int x^2 P(x) \,dx = Var(x) + \big[\int x P(x) \,dx\big] ^2, $$

which for $$P_{gt}$$ is:

$$ \int x^2 P_{gt} \,dx = \sigma_{gt}^2 + \mu_{gt} ^2 $$

Substituting in the equation above, we get the following:

$$ \int P_{gt} \log \big(P_{gt}\big) \,dx = -\log \big( \sigma_{gt} \sqrt{2\pi}  \big) -\frac{1}{2\sigma_{gt}^{2}} \big[\sigma_{gt}^2 + \mu_{gt} ^2- 2 \mu_{gt}^2  + \mu_{gt}^2 \big] $$

After further simplification we get the following for $$T1$$:

$$ \int P_{gt} \log \big(P_{gt}\big) \,dx = -\log \big( \sigma_{gt} \sqrt{2\pi}  \big) -\frac{1}{2} $$

### Focusing on term $$T2$$
Now we can start simplifying term $$T2$$:

$$ - \int P_{gt} \log \big(P_{m}\big) \,dx = - \int P_{gt} \log \big(\frac{1}{\sigma_{m} \sqrt{2\pi}} e ^ {-\frac{1}{2} \big[\frac{(x-\mu_{m})^2}{\sigma_{m}^{2}}\big]}\big) \,dx $$

We can expand the logarithm term and simplify similar to how we did with term $$T1$$ above.

$$  = - \int P_{gt} \log \big(\frac{1}{\sigma_{m} \sqrt{2\pi}}\big) \,dx - \int P_{gt} \log \big( e ^ {-\frac{1}{2} \big[\frac{(x-\mu_{m})^2}{\sigma_{m}^{2}}\big]}\big) \,dx $$

$$  = \log \big(\sigma_{m} \sqrt{2\pi}\big) \int P_{gt} \,dx + \int \frac{P_{gt}}{2} \big[\frac{(x-\mu_{m})^2}{\sigma_{m}^{2}}\big]\log \big( e \big) \,dx $$

$$  = \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \int (x-\mu_{m})^2 P_{gt} \,dx $$  

$$  = \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \int (x^2-2x\mu_{m}+\mu_{m}^2) P_{gt} \,dx $$  


$$  = \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \int x^2 P_{gt}\,dx -2 \mu_{m} \int x P_{gt}\,dx + \mu_{m}^2 \int P_{gt}\,dx  \big] $$  

$$  = \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \int x^2 P_{gt}\,dx -2 \mu_{m} \mu_{gt} + \mu_{m}^2  \big] $$  

$$ - \int P_{gt} \log \big(P_{m}\big) \,dx = \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \sigma_{gt}^2 + \mu_{gt} ^2 -2 \mu_{m} \mu_{gt} + \mu_{m}^2  \big] $$  

$$ - \int P_{gt} \log \big(P_{m}\big) \,dx = \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \sigma_{gt}^2 + (\mu_{gt}-\mu_{m})^2  \big] $$  

### Substituting terms $$T1$$ and $$T2$$ in original equation
Recall:

$$ D_{KL}(P_{gt} \parallel P_{m}) = \underbrace{\int P_{gt} \log \big(P_{gt}\big) \,dx}_{T1} - \underbrace{\int P_{gt} \log \big(P_{m}\big) \,dx}_{T2} $$

We can substitute the terms $$T1$$ and $$T2$$ in this equation:

$$ D_{KL}(P_{gt} \parallel P_{m}) = -\log \big( \sigma_{gt} \sqrt{2\pi}  \big) -\frac{1}{2} + \log \big(\sigma_{m} \sqrt{2\pi}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \sigma_{gt}^2 + (\mu_{gt}-\mu_{m})^2  \big] $$

$$ D_{KL}(P_{gt} \parallel P_{m}) = -\frac{1}{2} + \log \big(\frac{\sigma_{m}}{\sigma_{gt}}\big) + \frac{1}{2 \sigma_{m}^{2}} \big[ \sigma_{gt}^2 + (\mu_{gt}-\mu_{m})^2  \big] $$


### Relation to L2 loss
Now imagine that for our task the standard deviation of the distributions are always fixed to be $$1$$, i.e. $$\sigma_{gt} = \sigma_{m}=1$$. In this case the above equation can be further simplified:

$$ D_{KL}(P_{gt} \parallel P_{m}) = -\frac{1}{2} + \log \big(1\big) + \frac{1}{2} + \frac{1}{2} \big[ (\mu_{gt}-\mu_{m})^2  \big] $$

$$ D_{KL}(P_{gt} \parallel P_{m}) = \frac{1}{2} \big[ (\mu_{gt}-\mu_{m})^2  \big] $$

Does the last equation look familiar? That's right! That is the equation for L2 loss!

-----------------------

In the next post, I plan to explore a bit more into what are the different metrics for learning Gaussian-based distributions and how they can be used in ML. I promise it will have less math, more code! 

+++
title = 'Deriving cross-entropy losses, from binary classification to LLM distilation'
date = 2025-01-15T23:36:23+02:00
draft = false
+++

## Binary logistic regression, binary cross-entropy loss 

We start with the binary logistic regression, and define our task as follows.

We're given a dataset inputs and targets (labels):

$$
\begin{array}{}
(x^{(1)}, y^{(1)})\\
(x^{(2)}, y^{(2)})\\
\vdots\\
(x^{(m)}, y^{(m)})
\end{array}
$$

and a _logistic model_:

$$\hat{y}^{(i)} = h(f(x^{(i)}; \theta)) = \frac{1}{1 + e^{-f(x^{(i)}; theta)}}$$

Each target \(y^{(i)}\) is either 0 or 1. That is, were doing binary classification, for example fraud/not fraud, churn/not churn, disease/not disease, cat/dog.

By \(f(x^{(i)}, \theta)\) we denote a function of \(x\) parametrized by \(\theta\). In case of neural networks, \(f\) is the network architecture, \(x\) are network inputs and \(\theta\) are the values of network parameters (*weights*).

Also note that \(h\) is the [logistic function](https://en.wikipedia.org/wiki/Logistic_function). Here we apply it to the result of \(f(x, \theta)\), so \(h(f(x, \theta))\) is equivalent to PyTorch `torch.sigmoid(model(x))`


Let's interpret the output of the model, \(\hat{y}^{(i)}\), as the estimated probability that \(y_i=1\), that is:

$$
\displaylines{
\begin{align}
P(y^{(i)}=1|x^{(i)};\theta) &= \hat{y}^{(i)} \\
P(y^{(i)}=0|x^{(i)};\theta) &= 1 -  \hat{y}^{(i)}
\end{align}
}
$$

or more succintly:

$$
\begin{array}{rl}
P(y^{(i)}|x^{(i)};\theta) = & {\hat{y}^{(i)}}^{y^{(i)}}\cdot(1-\hat{y}^{(i)})^{(1-y^{(i)})} \\
= & \begin{cases}
\hat{y}^{(i)} & \text{if } y^{(i)} = 1 \\
1-\hat{y}^{(i)} & \text{if } y^{(i)} = 0
\end{cases} \\
\end{array}
$$


For logistic regression, our goal is to find the values for \(\theta\) which maximize the joint probability of our dataset under the model distribution. Expressed as the function of model parameters \(\theta\), we call this joint probability the **likelihood** function.

The likelihood of the model \(f(x; \theta)\) given a dataset \(X, y\) is:

$$
\mathcal{L}(\theta;X, y) = \prod_{i=1}^{m}P(y^{(i)}|x^{(i)};\theta)=\prod_{i=1}^{m}{{\hat{y}^{(i)}}^{y^{(i)}}\cdot(1-\hat{y}^{(i)})^{(1-y^{(i)})}}
$$

When training the model we're looking for \(\theta_{MLE}\) that maximizes \(\mathcal{L}\):

$$\theta_{MLE} = \underset{\theta}{\mathrm{argmax}}(\mathcal{L}(\theta; X, y))$$

Since \(\mathcal{L}\) is a product of probabilities, it is always \(\ge0\), so we can look for the max of its logarithm:

$$\theta_{MLE} = \underset{\theta}{\mathrm{argmax}}\{\mathcal{L}\} = \underset{\theta}{\mathrm{argmax}}\{\log\mathcal{L}\}$$

Let's call \(\log\mathcal{L}\) *log-likelihood* and denote it as \(\mathscr{l}\). Applying properties of logarithms we have:

$$
\begin{align*}
\mathscr{l}(\theta) &= \log\mathcal{L}(\theta) \\
&= \log\prod_{i=1}^{m}{{\hat{y}^{(i)}}^{y^{(i)}}\cdot(1-\hat{y}^{(i)})^{(1-y^{(i)})}} \\
&= \sum_{i=1}^{m}{\log\left\{{\hat{y}^{(i)}}^{y^{(i)}}\cdot(1-\hat{y}^{(i)})^{(1-y^{(i)})}\right\}} \\
&= \sum_{i=1}^{m}{ y^{(i)}\log\hat{y}^{(i)} } + (1 - y^{(i)})\log(1-\hat{y}^{(i)})
\end{align*}
$$

So our optimal set of parameters \(\theta_{MLE}\) is the one that **maximizes** the log-likelihood. In the context of machine learning, we usually speak of **minimizing** some loss. Thus, in order to turn it into a loss, we'll just take the negative of the log likelihood:

$$
J(\theta) = -\mathscr{l}(\theta) = -\sum_{i=1}^{m}{ y^{(i)}\log\hat{y}^{(i)} } + (1 - y^{(i)})(1-\hat{y}^{(i)})
$$

which gives us the familar form of the [**binary cross entropy loss**](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html).

## Multi-class classification, softmax, cross-entropy loss

We have a multi-class classification problem when our targets \(y_i\) are not binary, but denote one out of \(C\) classes. We give two examples of this:

1. Given an image, the model should say if it's a photo of a cat, dog, horse or cow. The targets are given as cat: \(y=0\), dog: \(y=1\), cow: \(y=2\), horse: \(y=3\).
2. Given an input sequence of words, coming from a _vocabulary_ \(V\) (vocabulary = the set of all possible words), predict which word continues the sequence. This is the well known problem of language modeling.
    
    The GPT family of large language models solves this problem in a way that is equivalent to multi-class classification. For some input sequence \(x\), for each token \(w_i \in V\), we're modeling the probability \(P(w_i|x)\) that \(w_i\) is the next token.

Expanding on our approach to binary classification, let's have a separate model \(f_j(x;\theta_j)\) for each class \(j=1\ldots C\). Since we want to model a multinomial distribution - each target \(y^{(i)}\) is one of \(C\) possible options - we'd like our probability estimates to sum up to 1.

If we converted each \(f_j(x;\theta)\) into a probability using the sigmoid function as we did for the binary case, we wouldn't be able to guarantee this property.

Instead, we'll use the vector function **softmax**, defined as

$$
\mathrm{softmax}(t_1, t_2, \ldots, t_C) = \left[
\begin{array}{}
\frac{\exp(t_1)}{\sum_{j=1}^{C}{\exp(t_j)}} \\
\vdots \\
\frac{\exp(t_C)}{\sum_{j=1}^{C}{\exp(t_j)}} 
\end{array}
\right]
$$

Note that output of the softmax has the property that it's entries are in the interval \([0, 1]\), and they sum up to 1, just as we need for our probability model.

Now our multiclass probability model can be written as:

$$
\begin{align*}{}
\hat{y}^{(i)}=
\left[
\begin{array}{}
\hat{P}(y^{(i)}=1|x;\theta) \\
\vdots \\
\hat{P}(y^{(i)}=C|x;\theta) \\
\end{array}
\right]
= \mathrm{softmax}(f_1(x^{(i)};\theta_1), \ldots, f_C(x^{(i)};\theta_C)) 
= \left[
\begin{array}{}
\frac{\exp(f_1(x^{(i)};\theta_1))}{\sum_{j=1}^{C}{\exp(f_j(x^{(i)}; \theta_j))}} \\
\vdots \\
\frac{\exp(f_C(x^{(i)};\theta_1))}{\sum_{j=1}^{C}{\exp(f_j(x^{(i)}; \theta_j))}} \\
\end{array}
\right]
\end{align*}
$$

How do we optimize this? Again, let's look at the negative log-likelihood:

$$
\begin{align*}
J(\theta) = -\mathscr{l}(\theta)
& = -\log\left\{ \prod_{i=1}^{m}\hat{P}(y^{(i)}|x^{(i)};\theta) \right\} \\
& = -\log\left\{ \prod_{i=1}^{m}{\hat{y}^{(i)}_{y^{(i)}} }\right\} \\
& = -\log\left\{ \prod_{i=1}^{m}{\mathrm{softmax}(f_1(x^{(i)};\theta_1), \ldots, f_C(x^{(i)};\theta_C))_{y^{(i)}} }\right\} \\
& = -\sum_{i=1}^{m}\log\left\{
  \frac{\exp(f_{y^{(i)}}(x^{(i)};\theta_{y^{(i)}}))}{\sum_{j=1}^{C}{\exp(f_j(x^{(i)}; \theta_j))}}
\right\} \\
\end{align*}
$$

Let's map this to neural networks. To do multi-class classificaton, we'll have a classification head that's usually a linear layer with \(C\) output features, that we call *logits*. Applying the softmax to the output vector of this layer, we get the predicted class probabilities.

Note that the loss equation above corresponds to the [cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.htmlhttps://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) which we typically use to train multiclass classification models. The PyTorch implementation takes as inputs the *logits*, which correspond to \(f_j\) in our equations.


## Entropy and cross-entropy view, equivalence of KL-divergence minimization and maximum likelihood

The *Shannon entropy* of a discrete distribution \(P_X\) of a random variable \(X\) is defined as 

$$
H(P_X) = - \sum_{x\in X}P_X(x)\cdot\log P_X(x)
$$

The motivation for the formula above is as follows:
- We'd like to measure the amount of *surprise* when we seen an outcome \(x\) of a random variable \(X\)
- The less probable the outcome is, the more surprised we are to see it
- We want the amount of surprise upon seeing two events \(x_1 \cap x_2\) to be the sum of the surprises of each event

The function that satisfies these conditions is \(-log(P(x))\). Given this, if we want to quantify the uncertainty or the amount of disorder (entropy) that a probability distribution carries, we take the entropy of the random variable to be the expected surprise. This yields the definition above.

Now assume we have a true distribution \(P_X\) and an estimated (model) distribution \(Q_X\). If we're observing outcomes of \(P_X\) thinking that they are coming from \(Q_X\), the expected *surprise* is given by:

$$
H(P_X, Q_X) = - \sum_{x\in X}P_X(x)\cdot\log Q_X(x)
$$

We'll call \(H(P_X, Q_X)\) the **crossentropy** of \(P_X\) and \(Q_X\). (Note that crossentropy is not symmetrical, ie. \(H(P_X, Q_X) \neq H(Q_X, P_X)\))

Having defined entropy and cross-entropy, we can try to quantify *how well* \(Q_X\) models \(P_X\). For that, we use *relative entropy* or *Kullback-Leibler divergence* between \(P\) and \(Q\):

$$
\begin{align*}
KL(P||Q) & = \left(-\sum_{x\in X}P(x)\cdot\log Q(x)\right) - \left( - \sum_{x\in X}P(x)\cdot\log P(x) \right) \\
& = H(P_X, Q_X) - H(P_X) \\
& = \sum_{x\in X}P(x)\cdot\log \frac{P(x)}{Q(x)}
\end{align*}
$$

What's the intuition behind relative entropy? If we expect random events \(x_1, x_2, \ldots\) to come from \(Q_X\), when in fact they come from \(P_X\), the expected surprise is given by the first term \(\left(-\sum_{x\in X}P(x)\cdot\log Q(x)\right)\), which is \(H(P_X, Q_X)\) . The expected surprise we'd see if we knew the true distribution is given by the second term, which is exactly \(H(P_X)\). Thus, \(KL(P||Q)\) tells us the difference in the expected surprise when we think \(X\) is distributed as \(Q_X\), but in fact its true distribution is \(P_X\).

It can be shown that \(KL(P||Q) \ge 0\), and that \(KL(P||Q) = 0\) only when \(P_X = Q_X\). Building on this, another useful interpretation is that KL-divergence measures how well the model distribution approximates the true distribution. That is, it measures how *close* some two distributions are. 

Taking this to our softmax model, our goal is to approximate the true distribution \(P_{Y|X}(y|x)\) by building a model \(\hat{P}_{Y|X}(y|x;\theta)\) (probability of \(y\) conditined on \(x\) and parametrized by \(\theta\)). To measure how close the model distribution is to the true distribution, let's look at the KL-divergence:

$$
\begin{align*}
KL(P_{X|Y}||\hat{P}_{X|Y};\theta) & =
  \left(
    -\sum_{j=1}^{C}{P(y=j|x)\log\hat{P}(y=j|x;\theta)}
  \right)
  -\left(
    -\sum_{j=1}^{C}{P(y=j|x)\log P(y=j|x)}
  \right) \\
& = \sum_{j=1}^{C}{P(y=j|x)\log\frac{P(y=j|x)}{\hat{P}(y=j|x;\theta)}}
\end{align*}
$$

or for the joint distribution \(P_{X, Y}\):

$$
\begin{align*}
KL(P_{X,Y}||\hat{P}_{X,Y};\theta) & =
  \left(
    -\sum_{x, y\in \mathcal{X}\times\mathcal{Y}}{P_{X, Y}(y,x)\log\hat{P}_{X, Y}(y,x;\theta)}
  \right)
  -\left(
    -\sum_{x, y\in \mathcal{X}\times\mathcal{Y}}{P_{X, Y}(y,x)\log P_{X, Y}(y,x;\theta)}
  \right) \\
& = \sum_{x, y\in \mathcal{X}\times\mathcal{Y}}{P_{X, Y}(y,x)\log\frac{P_{X,Y}(y, x)}{\hat{P}_{X, Y}(y, x;\theta)}} \\
& = \sum_{x, y\in \mathcal{X}\times\mathcal{Y}}{P_{X, Y}(y,x)\log\frac{P(y|x)\cancel{P(x)}}{\hat{P}(y|x;\theta)\cancel{P(x)}}} \\
& = E_{X,Y}\left[ \log\frac{P(y|x)}{\hat{P}(y|x;\theta)}\right]
\end{align*}
$$

Now, if we observe a sequence of \(n\) iid. random samples of \(X, Y\), the law of large numbers (LLN) tells us that:

$$
\frac{1}{n}\sum_{i=1}^{n}\log\frac{P(y^{(i)}|x^{(i)})}{\hat{P}(y^{(i)}|x^{(i)};\theta)} \xrightarrow[n\to\infty]{}E_{X,Y}\left[ \log\frac{P(y|x)}{\hat{P}(y|x;\theta)}\right] = KL(P_{X,Y}||\hat{P}_{X,Y};\theta)
$$

This gives us a way to estimate \(KL(P_{X,Y}||\hat{P}_{X,Y};\theta)\) given a dataset of \(X, y\) samples.

Further, if we know that the best approximating distribution is the one that minimizes the KL-divergence, the way to infer the optimal model distribution from the dataset is:

$$
\underset{\theta}{\mathrm{argmin}}\left\{ \frac{1}{n}\sum_{i=1}^{n}\log\frac{P(y^{(i)}|x^{(i)})}{\hat{P}(y^{(i)}|x^{(i)};\theta)} \right\} 
\xrightarrow[n\to\infty]{}
\underset{\theta}{\mathrm{argmin}}\left\{KL(P_{X,Y}||\hat{P}_{X,Y};\theta)\right\}
$$

$$
\begin{align*}
\underset{\theta}{\mathrm{argmin}}\left\{ \frac{1}{n}\sum_{i=1}^{n}\log\frac{P(y^{(i)}|x^{(i)})}{\hat{P}(y^{(i)}|x^{(i)};\theta)} \right\}
& = \underset{\theta}{\mathrm{argmax}}\left\{ \frac{1}{n}\sum_{i=1}^{n}\log{\hat{P}(y^{(i)}|x^{(i)};\theta)} \right\} \\
& =  \underset{\theta}{\mathrm{argmax}}\left\{ 
  \frac{1}{n}\log \prod_{i=1}^{n}{\hat{P}(y^{(i)}|x^{(i)};\theta)}
  \right\} \\
& =  \underset{\theta}{\mathrm{argmax}}\left\{ 
  \frac{1}{n}\mathscr{l}(\theta)
  \right\} \\
\end{align*}
$$

where \(\mathscr{l}(\theta)\) is the log-likelihood function that we defined at the beginning.

So we started with a view based on information theory, we looked for a set of parameters \(\theta\) that yields a model distribution which minimizes the Kullback-Leibler divergence between the actual distribution and the model, and we ended up with the same optimal \(theta\) that the maximum likelihood method gave us.

There's one thing left to clarify. When we gave the final loss function for the maximum likelihood estimates, we said they're the familiar forms of **binary cross-entropy** and **cross-entropy** loss. We haven't made it obvious yet why these terms are actually called cross-entropy losses.

To see why, note that KL can be written as the difference between a crossentropy and an entropy term. Since the entropy part isn't a function of \(\theta\), the optimal \(\theta\) minimizes KL by minimizing just its cross-entropy term:

$$
\begin{align*}
\underset{\theta}{\mathrm{argmin}}\left\{
  KL(P_{X,Y}||\hat{P}_{X,Y};\theta)
\right\} &= 
\underset{\theta}{\mathrm{argmin}}\left\{
  \underbrace{H(P_{X,Y}, \hat{P}_{X,Y})}_\text{cross-entropy} - \underbrace{H(P_{X, Y})}_\text{entropy}
\right\} \\
& = \underset{\theta}{\mathrm{argmin}}\left\{
  H(P_{X,Y}, \hat{P}_{X,Y})
\right\} \\
\end{align*}
$$

Here we conclude that maximizing the log-likelihood, minimizing the Kullback-Leibler divergence and minimizing the cross-entropy are equivalent and all yield the same estimated parameters.

## The special case of model distilation

Model distilation (or knowledge distilation) is a special case of model training where we are teaching a student model \(M^{student}\) (typically smaller) to match the outputs of the teacher model (typically larger) \(M^{teacher}\).

When training a model from scratch, although our targets come from a distribution \(P_Y\), we never know the actual distribution. We only see the inputs \(x^{(i)}\) and the targets \(y^{(i)}\), which we assume to be individual samples from \(P_Y\).

Now, when we have an already trained teacher model, we take its outputs as the ground truth. If the teacher model outputs the conditional probabilities \(P^{teacher}(y|x)\), we take those as the ground-truth. That is, we assume that the output probabilities of \(M^{teacher}\) represent the true distribution \(P_{Y|X}(y|x)\)

Another case where we have a full distribution instead of *hard* class labels as targets, is *label smoothing*. Basically, instead of saying _"this example is 100% class \(i\)"_, we say _"this example is \(\epsilon\%\) class \(i\), and \((100-\epsilon)/(k-1)\%\) any of the other \(k-1\) classes"_.

Let's now see how we can build this additional information into our loss function.

We've shown above that in order to find the model distribution that best approximates the true distribution, we need to find the \(\theta\) that minimizes the KL-divergence between the true and model distribution. And to minimize the KL-divergence, we need to minimize its cross-entropy term, so we'll start from there.


$$
\begin{align*}
\underset{\theta}{\mathrm{argmin}}\left\{
  KL(P_{X,Y}||\hat{P}_{X,Y};\theta)
\right\} &= 
\underset{\theta}{\mathrm{argmin}}\left\{
  H(P_{X,Y}, \hat{P}_{X,Y})
\right\} \\
&= \underset{\theta}{\mathrm{argmin}}\left\{
    -\sum_{x, y\in \mathcal{X}\times\mathcal{Y}}{P_{X, Y}(y,x)\log\hat{P}_{X, Y}(y,x;\theta)}
  \right\} \\
&= \underset{\theta}{\mathrm{argmin}}\left\{
    -\sum_{x\in \mathcal{X}}\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x)P_{X}(x)\log\hat{P}_{Y|X}(y|x;\theta)}P_X(x)
  \right\} \\
&= \underset{\theta}{\mathrm{argmin}}\left\{
    -\sum_{x\in \mathcal{X}}P_X(x)\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x)\left( \log\hat{P}_{Y|X}(y|x;\theta)+\log P_X(x) \right)}
  \right\} \\
&= \underset{\theta}{\mathrm{argmin}}\left\{
      -\sum_{x\in \mathcal{X}}P_X(x)\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x)\log\hat{P}_{Y|X}(y|x;\theta)
      -\underbrace{\sum_{x\in \mathcal{X}}P_X(x)\sum_{y\in \mathcal{Y}}{P_{Y|X}(y,x)\log P_{X}(x)}}_{\text{not a function of }\theta}
    }
  \right\} \\
&= \underset{\theta}{\mathrm{argmin}}\left\{
      -\sum_{x\in \mathcal{X}}P_X(x)\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x)\log\hat{P}_{Y|X}(y|x;\theta)
    }
  \right\} \\
&= \underset{\theta}{\mathrm{argmin}}\left\{
      -E_X\left[ \sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x)\log\hat{P}_{Y|X}(y|x;\theta)} \right]
  \right\} \\
\end{align*}
$$

Before continuing, we need to note two important things that went our way here:
- We're able to write \(\hat{P}_{X, Y}(x, y;\theta) = \hat{P}_{Y|X}(y|x;\theta)P_X(x) \) because we're only modeling the conditional part of the distribution, and \(P_X(x)\) is whatever the distribution of input samples we get in the dataset.
- Note the problem setup - our dataset will consist of input samples \(x^{(1)}, \ldots, x^{(n)}\), but the targets won't be samples of \(Y\). Instead our targets are the entire *"true"* conditional distributions of \(Y\). This is important to make the law of large numbers work our way, as in the next step we'll estimate \(E_x\) for which we need iid. samples of \(X\).

$$
\begin{align*}
\underset{\theta}{\mathrm{argmin}}\left\{
      -E_X\left[ \sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x)\log\hat{P}_{Y|X}(y|x;\theta)} \right]
  \right\} \xrightarrow[n\to\infty]{}
  \underset{\theta}{\mathrm{argmin}}\left\{
    -\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x^{(i)})\log\hat{P}_{Y|X}(y|x^{(i)};\theta)}
  \right\}
\end{align*}
$$

Let's plug in our softmax model here. First we recall that for a softmax model we defined:

$$
\hat{P}_{Y|X}(y|x;\theta) = \frac{\exp(f_y(x;\theta_y))}{\sum_{j=1}^{C}{\exp(f_j(x; \theta_j))}} \\
$$

Plugging this into our loss function, we get:

$$
\begin{align*}
J(\theta) & = 
  -\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x^{(i)})\log\hat{P}_{Y|X}(y|x^{(i)};\theta)}\\
& = -\frac{1}{n}\sum_{i=1}^{n}\sum_{y\in \mathcal{Y}}{P_{Y|X}(y|x^{(i)})\log\
\frac{\exp(f_y(x{(i)};\theta_y))}{\sum_{j=1}^{C}{\exp(f_j(x^{(i)}; \theta_j))}}
}\\
\end{align*}
$$

With a slightly different notation, this is exactly PyTorch's **mean-reduced cross entropy loss when the targets are the probabilities for each class**. From [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html):


> *The target that this criterion expects should contain either:*
> - ...
>
> - _**Probabilities for each class** ; useful when labels beyond a single class per minibatch item are required, such as for blended labels, label smoothing, etc. The unreduced (i.e. with reduction set to 'none') loss for this case can be described as:_
>
> $$
\mathscr{l}(x, y) = L = {l_1, \ldots, l_N}^{T},\quad l_n = -\sum_{c=1}^{C}w_c\log\frac{exp(x_{n,c})}{\sum_{i=1}^{C}exp(x_{n, i})}$$
>
> _where \(x\) is the input, \(y\) is the target, \(w\) is the weight, \(C\) is the number of classes, and \(N\) spans the minibatch dimension as well as \(d_1, \ldots, d_k\) for the K-dimensional case. If `reduction` is not `none` (default `mean`), then_
>
> $$
\mathscr{l}(x,y) = \begin{cases}
\frac{\sum_{n=1}^{N}l_n}{N},\quad \text{if reduction = 'mean'}\\
\sum_{n=1}^{N}l_n,\quad \text{if reduction = 'sum'}\\
\end{cases}
$$





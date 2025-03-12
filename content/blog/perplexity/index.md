+++
title = 'The Intuition and Derivation of Perplexity for LLM Evaluation'
date = 2025-03-08T10:36:23+02:00
draft = false
+++

## The intuition

When [deriving the cross-entropy loss](/blog/cross-entropy-loss), we've shown how entropy plays a central role in the optimization of softmax models (ie. multi-class classification models). 

All large language models (LLMs) are exactly that - softmax models that for an input sequence of \(t\) tokens \(x=[x_1, x_2, \ldots, x_t]\) output a conditional probability distribution \(P(w|x)\) over the vocabulary \(V\) of all tokens. This distribution gives us the most likely next token(s) to continue the input sequence.

Consider an example:

$$
\underbrace{\text{The cat sat on the}}_{x_{1..t}}\ \  \underbrace{\text{...}}_{x_{t+1}} \quad\begin{cases}
\begin{align*}
\quad\vdots \quad \\
\text{couch}\quad &| \quad p=0.4\\
\text{mat}\quad &|\quad p=0.5\\
\text{moon}\quad &|\quad p=0.0001\\
\quad\vdots\quad \\
\end{align*}
\end{cases}
$$

The most likely next word in the sentence "_The cat sat on the ..._" is "_mat_", and it's very unlikely that it is the word "_moon_". Note that for simplicity we use _words_ instead of _tokens_ here, although for modern LLMs a token is usually not a full word.

More formally, for some language model \(M\), we would denote its output (a vector) as \(\hat{y} = M(x)\) and say 

$$
\begin{align*}
\hat{y}_{\text{mat}} &= M(\text{The cat sat on the})_{mat} = P_M(\text{mat}\ |\ \text{The cat sat on the}) = 0.5 \\
\hat{y}_{\text{moon}} &= M(\text{The cat sat on the})_{moon} = P_M(\text{moon}\ |\ \text{The cat sat on the}) = 0.0001
\end{align*}
$$

Now instead of generating a new sequence, what if we take some sequence up to the last token \(x_{1..t-1}=[w_1, w_2, \ldots, w_{t-1}]\)  and feed it as input to the model? The output distribution tells us what the model thinks should be the next token \(x_t\). So what does the model say about the probability of \(x_t\) being exactly the \(w_t\) from our original sequence?

That is, in the following example:

$$
\underbrace{\text{John went to the store. He bought }}_{x_{1..t-1}}\ \  \underbrace{\text{milk}}_{x_{t}} 
$$

we ask what does the model \(M\) say is the probability that _"milk"_ continues the sentence _"John went to the store. He bought "_.

A good model of English would give _"milk"_ high probability. But let's say the output probability for this word, \(\hat{y}_{milk} = M(x_{1..t-1})_{milk}\), is very small, eg. \(0.00001\). We would be very surprised to see this model generate such a sequence. On the other hand, if \(\hat{y}_{w}\) for some word \(w\) is large, say \(0.9\), the model thinks that \(w_t\) is a very natural continuation of the sequence \(x_{1..t-1}\).

Here we can do a convenient personification - we could say that _the model itself_ would be very _surprised_, to see this word here. So if we feed the model some large number of English sentences, and it's very _surprised_ to see each new word, we would conclude that it's a bad model. If we test another model and it's less perplexed with our input dataset, we'd say it's a better (more accurate) language model.

How do we formalize and quantify this?

## Formalization. Cross-entropy

When [motivating the definition of entropy](/blog/cross-entropy-loss/#entropy-and-cross-entropy-view-equivalence-of-kl-divergence-minimization-and-maximum-likelihood), we talk about quantifying the _surprise_ with the outcome a random variable. We can do the same here.

Let's say our measure of surprise, according to a model \(M\) upon seeing \(x_t\) after \([x_1, \ldots, x_{t-1}]\) is:

$$
h(x_t|x_1,\ldots,x_{t-1}) = -\log P_M(x_t|x_1,\ldots,x_{t-1}) = -\log\hat{y}_{w_t}
$$


Now, what if we give the model a very long text or a dataset of sentences, and measure \(h(w)\) for every token? Summing these up would give us the total surprise of the model seeing this particular dataset. 

Say we feed a French text to a model trained only on English.
It would be very surprised by the sequence of tokens.[^1]
What if feed it two different books in French, one 100 pages long, the other 800 pages.
The total \(h\) would probably be about 8x larger for the longer text.

[^1]: Assuming the model is built on top of a vocabulary containing both English and French words.
Modern byte-pair encoding tokenizers are built in a language-agnostic way, so a model trained only on English texts can take input in French as well.

If we normalize the sum by the total number of input tokens, we would expect to get roughly equal values for the two texts. So let's define our _model-text surprise_ as:

$$
\begin{align*}
H_{M}(x)& = \frac{1}{t}\sum_{i=1}^{t}{h_M(x_{i}|x_0\ldots x_{i-1})} \\
&=\frac{1}{t}\sum_{i=1}^{t}{-\log\hat{y}_{x_t|x_0\ldots x_{i-1}}}
\end{align*}
$$

where \(M\) is our model, \(t\) is the length of the text, \(w_i\) is the i-th token of the text and \(w_0\) is the initial token (usually `BOS`). In practice, our models won't have infinite context, so for each token we'll look at up to _maximum context size_ previous tokens.

Let's call \(H_M(x)\) the __average per-token cross entropy__.

Now we'll proceed to show that this is actually an estimator of the cross-entropy between the actual language of the text and the model.

Let's assume we have a set of \(n\) sequences \(x^{(1)}, \ldots, x^{(n)}\), of lengths \(t^{(1)},\ldots, t^{(n)}\); each sequence can have a different size. They all come as a random sample from some language \(L\), defined over a vocabulary of tokens \(V\). We'll define \(L\) to be a discrete probability distribution \(P_L\) over \(\mathcal{L}\), the set of all possible token sequences of all lengths up to some finite maximum length \(T\), ie. the powerset \(\mathcal{P}(V^T)\). For convenience, we don't allow infinite sequences in our language, but \(T\) can be any finitely large value.

We also have a model \(M\) of that language, which for any input sequence \(x\in \mathcal{L}_{T-1}\) (ie. up to \(T-1\) tokens long) outputs a probability distribution \(P_M(w|x): V\rightarrow [0,1]\). That is, for an input sequence, the model outputs probabilities for the next token.

Let's look at our _model-text surprise_ measure, \(H_M\), over this set. For notational convenience, we will denote the last token in a sequence as \(x_{-1}\) and all of the sequence up to the last token as \(x_{\colon-1}\). We have:

$$
\begin{align*}
H_M(x^{(1)}, \ldots, x^{(n)}) & = \frac{1}{n}\sum_{i=1}^{n}{-\log\hat{y}_{x^{(i)}_{-1}|x^{(i)}_{\colon-1}}} \\
& = \frac{1}{n}\sum_{i=1}^{n}{-\log P_M(x^{(i)}_{-1}|x^{(i)}_{\colon-1})}  \xrightarrow[n\to\infty]{LLN}{}
  E_{x\in \mathcal{L}}\left[ -\log {P}_M(x_{-1}|x_{\colon-1})\right] \\
&= - \sum_{x\in \mathcal{L}}P_L(x)\log{P}_M(x_{-1}|x_{\colon-1}) \\
&= - \sum_{x\in \mathcal{L}}P_L(x_{-1}|x_{:-1})P_L(x_{:-1})\log{P}_M(x_{-1}|x_{\colon-1}) \\
&= - \sum_{x_{:-1}\in \mathcal{L}_{T-1}}P_L(x_{:-1})\left\{\sum_{x_{-1}\in V}P_L(x_{-1}|x_{:-1})\log{P}_M(x_{-1}|x_{\colon-1}) \right\} \\
&= - \sum_{x\in \mathcal{L}_{T-1}}P_L(x)\left\{\sum_{w\in V}P_L(w|x)\log{P}_M(w|x) \right\} \\
&= - \sum_{x\in \mathcal{L}_{T-1}}P_L(x)H(P_{L_{w|x}}, P_{M_{w|x}}) \\
&= E_{\mathcal{L}_{T-1}}\left[H(P_{L_{w|x}}, P_{M_{w|x}})\right]
\end{align*}
$$

In the final term, \(H(P_{L_{w|x}}, P_{M_{w|x}})\) is the cross-entropy between the real and modeled conditional distribution of the next token \(w\) given context \(x\). \(E_{\mathcal{L}_{T-1}}\) is the expectation over all sequences of the language \(L\) that can be continued, ie. sequences of length less than the max defined length \(T\).

Thus, our inutitively defined surprise measure is an estimate of the expected per-token crossentropy between the actual language and our model [^2].

[^2]:Note that in language modeling literature we'll find a different derivation of the similar result, motivated by estimating the entropy of natural languages over infinite sequences. Since here we're mostly conncered with comparing the accuracy of different models over finite datasets, we might as well constrain our language to be defined over the set of all subsequences of a given dataset.

## From cross-entropy to perplexity. Interpretation of perplexity

Let's define **perplexity** as the exponential of the per-token cross-entropy:

$$
\mathrm{Perplexity}(M;x) = \exp\left\{H_M(x)\right\}
$$

Of course, it's important that we use the same exponental base as we do for the logarithm inside \(H_M(x)\).

This doesn't look like anything special yet. But let's consider a model \(M\) which always outputs discrete uniform distribution over some \(k\) tokens. Conveniently, we'll assume that our \(x_i\) is always among those tokens.
That is, our average cross-entropy becomes:

$$
\begin{align*}
H_{M}(x)& = \frac{1}{t}\sum_{i=1}^{t}{-\log\hat{y}_{x_t|x_0\ldots x_{i-1}}} \\
&= \frac{1}{t}\sum_{i=1}^{t}{-\log\frac{1}{k}} \\
&= -\log\frac{1}{k} \\
&= \log k
\end{align*}
$$

Let's look at the perplexity of this model:

$$
\begin{align*}
\mathrm{Perplexity}(M;x) &= \exp\left\{H_M(x)\right\} \\
&= \exp\left\{\log k\right\} \\
&= k
\end{align*}
$$

So, a model which at every step \(i\) would decide \(x_i\) at random among \(k\) tokens, has perplexity of exactly \(k\).

Now let us compare a model \(M_1\) with \(\mathrm{Perplexity}(M_1; x) = 10.2\) to a model \(M_2\) with \(\mathrm{Perplexity}(M_2; x) = 89.7\). Applying our _Uniform(k) model_ intuition, we see that \(M_1\) generates tokens from much narrower distribution than \(M_2\). That is to say, \(M_1\) generates tokens with less uncertainty than \(M_2\).

Thus, perplexity quantifies the average uncertainty of a model in relation to a sequence \(x\).

Why is perplexity reported more commonly as a model performance measure, rather than cross-entropy? Manning and Schütze give a hint to this in "Foundations of statistical natural language processing":

> In the speech recognition community, people tend to refer to perplexity
rather than cross entropy.
> [...]
> We suspect that speech recognition people prefer to report the larger
non-logarithmic numbers given by perplexity mainly because it is much
easier to impress funding bodies by saying that “we’ve managed to reduce perplexity from 950 to only 540” than by saying that “we’ve reduced cross entropy from 9.9 to 9.1 bits.”
>
> However, perplexity does also have an intuitive reading: a perplexity of _k_ means that you are as surprised on average as you would have been if you had had to guess between kequiprobable choices at each step.

Perplexity has lost some of its importance in language model evaluation since GPT-3 and the rise of highly capable large language models.
This is mainly because perplexity itself doesn't capture the higher level capabilities of these models - their reasoning power, factual accuracy, creativity, etc.

However, perplexity is still very useful as an easy to calculate proxy-benchmark.

For example, it is regularly used in quantization literature.
A good quantization procedure should keep perplexity of the quantized model as close as possible to that of the full-precision version. [Jin et al. 2024](https://arxiv.org/abs/2402.16775) further confirm that for quantized model, perplexity is strongly correlated with the perfomance on other benchmarks.

In "AI Engineering (O'Reilly, 2025)", Chip Huyen lists more use cases of perplexity:
- It's a proxy for downstream task performance
- A model is expected to have very low perplexity for a text from it's training data. Thus, perplexity can be used to detect if a model has been trained on a particular piece of data.
- Building training dataset - add new data to an existing dataset only if perplexity on it is relatively high. Remove texts that have extremely high perplexity values, as they are likely gibberish.


## References

[1] C. Manning, H. Schütze - Foundations of statistical natural language processing (1999, The MIT Press)

[2] D. Jurafsky, J. Martin - [Speech and Language Processing 3rd ed.](https://web.stanford.edu/~jurafsky/slp3/) (2025)

[3] C. Bishop - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) (Springer, 2006)

[4] C. Huyen - AI Engineering (O'Reilly, 2025)

[5] Jin et al. [A Comprehensive Evaluation of Quantization Strategies for Large Language Models](https://arxiv.org/abs/2402.16775) (2024)







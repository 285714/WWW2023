# Learning Mixtures of Markov Chains with Quality Guarantees

This repository contains an implementation of the method described in https://arxiv.org/abs/2302.04680. You can use it to learn mixtures of `L` Markov chains on `n` vertices given a set of 3-trails, i.e. state observations of the form `i ⟶ j ⟶ k`.
Experiments from the paper can be found in [a Jupyter notebook](experiments.ipynb).
See below how to use the code.

## Mixtures and Distributions

The file [data.py](data.py) contains two Python classes to construct mixtures and trail distributions.  A mixture is defined through starting and transition probabilities. 
```python
n = 10
L = 3
mixture = Mixture.random(n, L)
print(mixture.S) # starting probabilities
print(mixutre.Ms) # transition probabilities
```
We can obtain the (exact) 3-trail distribution from a mixture and create a (noisy) sample from it.
```python
distribution = Distribution.from_mixture(mixture, 3)
sample = distribution.sample(100000) # creating a distribution from 100000 sample trails
```

## Learning a mixture

There are three methods in [learn.py](learn.py) to learn mixtures:

* The new method `svd_learn_new`
* The method `svd_learn` described in https://theory.stanford.edu/~sergei/papers/nips16-mcc.pdf
* Expectation maximization `em_learn`

Each take as input the sample distribution, `n`, and `L`. Note that `svd_learn_new` is able to infer the nubmers of chains if we pass `L=None` as argument. A sample run could look as follows
```python
learned_mixture = svd_learn_new(sample, n, L)
learned_mixture_em = em_learn(sample, n, L)
```

## Evaluation

The code provides functionality to evaluate the quality of the learned mixture. We can get the total variation (TV) distance on the mixtures, or on the trail distribution:
```python
recovery_error = mixture.recovery_error(learned_mixture)
recovery_error_em = mixture.recovery_error(learned_mixture_em)
print(f"Recovery Error: SVD_NEW={recovery_error.5f}, EM={recovery_error_em.5f}")
learned_distribution = Distribution.from_mixture(learened_mixture, 3)
learned_distribution_em = Distribution.from_mixture(learened_mixture_em, 3)
trail_error = distribution.dist(learned_distribution)
trail_error_em = distribution.dist(learned_distribution_em)
print(f"Trail Error: SVD_NEW={trail_error.5f}, EM={trail_error_em.5f}")
```


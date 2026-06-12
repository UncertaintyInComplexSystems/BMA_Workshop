# Bayesian model averaging workshop

Welcome to the Github repository containing material for the Donders workshop on Bayesian model averaging in Python, with JAX.

The workshop is organized into three files, corresponding to tutorials 0, 1, and 2. For each tutorial there is also a 'teacher's copy', which contains my worked solutions. Of course, you'll learn most by trying for yourself before looking at these. 

## Running the notebooks

While running the notebooks and installing `bamojax` is fairly straightforward, installing `jax` and `jaxlib` can be a bit of a pain. I therefore recommend to work in Google Colab. For each notebook, create one code cell up top and add these lines:

```
# We manually install dependencies
!pip install numpyro==0.19.0 blackjax==1.2.5 jaxtyping

# Install bamojax, without re-evaluating its jax/jaxlib dependencies
!pip install git+https://github.com/UncertaintyInComplexSystems/bamojax#egg=bamojax --no-deps
```

This installs `bamojax` and its key dependencies, `numypro`, `blackjax`, and `jaxtyping`.

More information and many examples of models and analyses can be found on the [bamojax github repository](https://github.com/UncertaintyInComplexSystems/bamojax).
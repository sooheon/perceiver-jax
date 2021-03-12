# Perceiver - JAX (Flax)

Implementation of [Perceiver](https://arxiv.org/abs/2103.03206) in JAX and Flax.

# Install

```shell
pip install perceiver-jax
```

# Usage

```python
import jax

from perceiver_jax import Perceiver

model = Perceiver(
    n_fourier_features=6,
    depth=8,
    n_latents=512,
    latent_n_heads=8,
    latent_head_features=64,
    cross_n_heads=2,
    cross_head_features=128,
    attn_dropout=0.,
    ff_mult=4,
    ff_dropout=0.,
)

RNG = jax.random.PRNGKey(42)
input_batch = jax.random.normal(RNG, (1, 224 * 224, 3))

y, variables = model.init_with_output({'params': RNG, 'dropout': RNG}, input_batch)
```

# Acknowledgements

Thanks to [lucidrains](https://github.com/lucidrains/) and his PyTorch
[implementation](https://github.com/lucidrains/perceiver-pytorch/) on which this is heavily based.

# Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

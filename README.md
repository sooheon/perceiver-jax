# Perceiver - JAX (Flax)

Implementation of [Perceiver](https://arxiv.org/abs/2103.03206) in JAX and Flax.
Also includes [ReZero](https://arxiv.org/abs/2003.04887) in lieu of LayerNorm, given its 
empirical benefits for very deep Transformers.

# Install

```shell
pip install perceiver-jax
```

Be sure to also install the correct accelerated jaxlib for your hardware.

# Usage

```python
import jax

from perceiver_jax import Perceiver

model = Perceiver(
    n_fourier_features=6,
    depth=8,
    n_latents=512,  # if input length is much smaller than this, reconsider using this architecture
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

You'll notice the parametrization is slightly simpler than with PyTorch, as you can infer 
input feature dimension shapes in JAX.

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

```bibtex
@misc{bachlechner2020rezero,
      title={ReZero is All You Need: Fast Convergence at Large Depth}, 
      author={Thomas Bachlechner and Bodhisattwa Prasad Majumder and Huanru Henry Mao and Garrison W. Cottrell and Julian McAuley},
      year={2020},
      eprint={2003.04887},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

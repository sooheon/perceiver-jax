import jax.nn.initializers as init
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import linen as nn


def default(val, d):
    return val if val is not None else d


def fourier_encode(x: jnp.ndarray, num_encodings=4):
    x = jnp.expand_dims(x, -1)
    orig_x = x
    scales = 2 ** jnp.arange(num_encodings)
    x /= scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate([x, orig_x], axis=-1)
    return x


class FeedForward(nn.Module):
    mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=False):
        features = x.shape[-1]
        x = nn.Dense(features * self.mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic=deterministic)
        x = nn.Dense(features)(x)
        return x


class Attention(nn.Module):
    heads: int = 8
    head_features: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, context=None, mask=None, deterministic=False):
        h = self.heads
        dim = self.head_features * h

        q = nn.Dense(dim, use_bias=False)(x)
        k, v = nn.Dense(dim * 2, use_bias=False)(default(context, x)).split(2, axis=-1)

        q, k, v = map(
            lambda arr: rearrange(arr, "b n (h d) -> (b h) n d", h=h), (q, k, v)
        )
        sim = jnp.einsum("b i d, b j d -> b i j", q, k) * self.head_features ** -0.5
        attn = nn.softmax(sim, axis=-1)
        out = jnp.einsum("b i j, b j d -> b i d", attn, v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)

        out = nn.Dense(x.shape[-1])(out)
        out = nn.Dropout(self.dropout)(out, deterministic=deterministic)
        return out


class ReZero(nn.Module):
    @nn.compact
    def __call__(self, x):
        scale = self.param("scale", init.zeros, (1,))
        return scale * x


class Perceiver(nn.Module):
    n_fourier_features: int = 4
    depth: int = 2
    n_latents: int = 256
    latent_n_heads: int = 8
    latent_head_features: int = 64
    cross_n_heads: int = 2
    cross_head_features: int = 128
    ff_mult: int = 4
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        bs, dim = x.shape[0], x.shape[-1]
        latents = self.param(
            "latents", init.normal(), (self.n_latents, dim * self.ff_mult)
        )
        pos_emb = self.param(
            "pos_emb", init.normal(), (self.n_latents, dim * self.ff_mult)
        )
        latent = repeat(latents + pos_emb, "n d -> b n d", b=bs)

        x = fourier_encode(x, self.n_fourier_features)
        x = rearrange(x, "b n ... -> b n (...)")

        for i in range(self.depth):
            rz = ReZero(name=f"rezero_{i}")
            latent += rz(
                Attention(
                    name=f"cross_attn_{i}",
                    heads=self.cross_n_heads,
                    head_features=self.cross_head_features,
                    dropout=self.attn_dropout,
                )(latent, x)
            )
            latent += rz(
                FeedForward(
                    name=f"cross_ff_{i}",
                    mult=self.ff_mult,
                    dropout=self.ff_dropout,
                )(latent)
            )
            latent += rz(
                Attention(
                    name=f"latent_attn_{i}",
                    heads=self.latent_n_heads,
                    head_features=self.latent_head_features,
                    dropout=self.attn_dropout,
                )(latent)
            )
            latent += rz(
                FeedForward(
                    name=f"latent_ff_{i}",
                    mult=self.ff_mult,
                    dropout=self.ff_dropout,
                )(latent)
            )
        return latent

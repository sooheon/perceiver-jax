from setuptools import setup, find_packages

setup(
    name="perceiver-jax",
    packages=find_packages(),
    version="0.0.3",
    license="MIT",
    description="Perceiver - JAX",
    author="Sooheon Kim",
    author_email="sooheon.k@gmail.com",
    url="https://github.com/sooheon/perceiver-jax",
    keywords=[
        "artificial intelligence",
        "deep learning",
        "transformer",
        "attention mechanism",
    ],
    install_requires=[
        "einops>=0.3",
        "jax>=0.2.10",
        "flax>=0.3.2"
    ],
)

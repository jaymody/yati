# Yet Another Transformer Implementation
[Transformer](https://arxiv.org/pdf/1706.03762.pdf) implementation in [JAX](https://github.com/google/jax)

This implementation is solely based on the [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, and attempts to follow it's terminology and specifications as closely as possible.


### Usage
**Install Dependencies** (Note: Tested and developed on Python 3.9.10)

```bash
poetry install
```

**Run Tests**

```bash
poetry run pytest --jaxtyping-packages model,typeguard.typechecked tests.py
```
`--jaxtyping-packages model,typeguard.typechecked` enables runtime checking for type hinted functions in `model` (including shapes/dimensions). If the array types or shapes don't match up, an error along the lines of:
`TypeError: type of argument "x" must be jaxtyping.Float[ndarray, 'n']; got jaxlib.xla_extension.DeviceArray instead` will be thrown, see https://github.com/google/jaxtyping/issues/33.

**Run Train Script**

```bash
poetry run python train.py
```

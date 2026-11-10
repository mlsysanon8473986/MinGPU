"""Candidate models: which layer counts are searched and what each one needs.

Everything that defines a candidate model is here; the formulas of the loss law are in
:mod:`mingpu.planner.define_search_space.scaling_law`. From the target loss and the law:

1. the compute-optimal model ``l_C``, the layer count whose ``N(l)`` is closest to ``N_C``,
   with its ``D_req`` and ``C_req`` (:func:`compute_optimal_model`);
2. the layer counts around it, ``l_C - 32`` to ``l_C + 32`` (:func:`candidate_layers`);
3. for each one, ``N(l)``, ``D_req`` and ``C_req`` (:func:`define_candidate_models`).

"Compute-optimal" is the adjective of the paper: the model of least training compute.

The model family has one shape, sized by its number of layers. A model with ``l`` layers
has ``l`` attention heads of dimension 128, hidden size ``h = 128 l`` and an ungated
two-matrix MLP of width ``4 h``, the shape of GPT-3 175B (Brown et al. 2020, Table 2.1).
Its blocks hold ``12 l h**2 = 196608 l**3`` parameters and its untied input and output
embeddings ``2 V h = 256 V l``, so

    N(l) = 196608 l**3 + 256 V l.

Layer counts are multiples of 8, so that the number of heads stays divisible by
the tensor-parallel degrees 2, 4 and 8.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .scaling_law import ScalingLaw, UnreachableLoss

#: Dimension of one attention head.
HEAD_DIM = 128
#: Default vocabulary: the 128,000 tokens of Llama 3 plus its 256 special tokens.
DEFAULT_VOCAB = 128_256
#: Layer counts are multiples of this step.
LAYER_STEP = 8
#: Default half-width of the candidate set around the compute-optimal model.
DEFAULT_SPAN = 32


@dataclass(frozen=True)
class Model:
    """A model of the family, set by its number of layers and its vocabulary."""

    layers: int
    vocab: int = DEFAULT_VOCAB

    def __post_init__(self) -> None:
        if self.layers <= 0 or self.vocab <= 0:
            raise ValueError(f"layers and vocab must be positive: {self.layers}, {self.vocab}")

    @property
    def hidden(self) -> int:
        return HEAD_DIM * self.layers

    @property
    def heads(self) -> int:
        return self.layers

    @property
    def feedforward(self) -> int:
        return 4 * self.hidden

    @property
    def n_block_params(self) -> int:
        """Parameters of the Transformer blocks, ``12 l h**2`` (what Calculon models)."""
        return 12 * self.layers * self.hidden**2

    @property
    def n_params(self) -> int:
        """All parameters, embeddings included: ``N(l)``, the ``N`` of the loss law."""
        return self.n_block_params + 2 * self.vocab * self.hidden

    def application(self, seq_len: int) -> dict[str, Any]:
        """Calculon application of the model for sequences of ``seq_len`` tokens."""
        return {
            "hidden": self.hidden,
            "feedforward": self.feedforward,
            "seq_size": seq_len,
            "attn_heads": self.heads,
            "attn_size": HEAD_DIM,
            "num_blocks": self.layers,
        }


@dataclass(frozen=True)
class CandidateModel:
    """A model and what it needs to reach the target loss (None: it cannot reach it)."""

    model: Model
    d_req: float | None
    c_req: float | None

    @property
    def reaches_loss(self) -> bool:
        return self.d_req is not None


def _layers_closest_to(target_params: float, vocab: int, step: int = LAYER_STEP) -> int:
    """The multiple of ``step`` whose ``N(l)`` is closest to ``target_params``.

    With ``target_params = N_C`` this is ``l_C``. Ties go to the smaller model.
    """
    if target_params <= 0:
        raise ValueError(f"target_params must be positive: {target_params}")

    def n_params(layers: int) -> int:
        return Model(layers, vocab).n_params

    layers = step
    while n_params(layers + step) < target_params:
        layers += step
    below, above = layers, layers + step
    if abs(n_params(above) - target_params) < abs(n_params(below) - target_params):
        return above
    return below


def candidate_layers(center: int, span: int = DEFAULT_SPAN, step: int = LAYER_STEP) -> list[int]:
    """Layer counts from ``center - span`` to ``center + span`` in steps of ``step``.

    The default is the nine models from ``l_C - 32`` to ``l_C + 32`` (paper,
    Sec. 4.1); counts below ``step`` are left out.
    """
    if center % step or span % step:
        raise ValueError(f"center {center} and span {span} must be multiples of {step}")
    return [layers for layers in range(center - span, center + span + 1, step) if layers >= step]


def _candidate_model(
    layers: int, target_loss: float, law: ScalingLaw, vocab: int
) -> CandidateModel:
    """The model of ``layers`` layers with its ``D_req`` and ``C_req`` for ``target_loss``."""
    model = Model(layers, vocab)
    try:
        d_req = law.d_req(model.n_params, target_loss)
        c_req = law.c_req(model.n_params, target_loss)
    except UnreachableLoss:
        d_req = c_req = None
    return CandidateModel(model, d_req, c_req)


def compute_optimal_model(
    target_loss: float, law: ScalingLaw, vocab: int = DEFAULT_VOCAB
) -> CandidateModel:
    """``l_C``, the model whose ``N(l)`` is closest to ``N_C``, with its ``D_req`` and ``C_req``.

    Raises :class:`UnreachableLoss` when the target loss is not above the law's irreducible loss.
    """
    n_c, _ = law.compute_optimal(target_loss)
    return _candidate_model(_layers_closest_to(n_c, vocab), target_loss, law, vocab)


def define_candidate_models(
    layers: Iterable[int], target_loss: float, law: ScalingLaw, vocab: int = DEFAULT_VOCAB
) -> list[CandidateModel]:
    """Each layer count as a :class:`CandidateModel`, in the given order."""
    return [_candidate_model(count, target_loss, law, vocab) for count in layers]

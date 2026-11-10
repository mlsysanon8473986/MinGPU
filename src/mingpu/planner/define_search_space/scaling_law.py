"""Loss law: the loss a model of ``N`` parameters reaches after ``D`` tokens.

MinGPU uses the parametric form of Hoffmann et al. (2022),

    L(N, D) = E + A / N**alpha + B / D**beta,

with ``N`` the total number of parameters, embeddings included. Any law of this
form can be supplied; two are predefined:

``BESIROGLU`` (default)
    Besiroglu et al. (2024), arXiv 2404.10102v2, Eq. 3 / Table 1: the refit of
    the Chinchilla data for which Hoffmann's three methods agree.
``CHINCHILLA``
    Hoffmann et al. (2022) Eq. 10, with the unrounded constants that Besiroglu
    et al. report in their Eq. 4 (``E``, ``alpha``, ``beta``; Eq. 10 as printed
    is rounded).
"""

from __future__ import annotations

from dataclasses import dataclass


class UnreachableLoss(ValueError):
    """Raised when a model cannot reach the target loss with any number of tokens."""


@dataclass(frozen=True)
class ScalingLaw:
    """``L(N, D) = E + A / N**alpha + B / D**beta``."""

    name: str
    E: float
    A: float
    B: float
    alpha: float
    beta: float

    def loss(self, n_params: float, tokens: float) -> float:
        """Loss of a model of ``n_params`` parameters trained on ``tokens`` tokens."""
        return self.E + self.A / n_params**self.alpha + self.B / tokens**self.beta

    def d_req(self, n_params: float, target_loss: float) -> float:
        """Fewest tokens with which a model of ``n_params`` parameters reaches ``target_loss``.

        This inverts the law in ``D``. Raises
        :class:`UnreachableLoss` when ``target_loss <= E + A / n_params**alpha``,
        the loss the model would reach with unlimited tokens.
        """
        gap = target_loss - self.E - self.A / n_params**self.alpha
        if gap <= 0:
            raise UnreachableLoss(
                f"a model of {n_params:.4g} parameters cannot reach loss {target_loss} "
                f"under the {self.name} law (its loss with unlimited tokens is "
                f"{target_loss - gap:.6f})"
            )
        return (self.B / gap) ** (1 / self.beta)

    def c_req(self, n_params: float, target_loss: float) -> float:
        """Training compute ``6 N D_req`` (FLOPs) of a model of ``n_params`` parameters."""
        return 6 * n_params * self.d_req(n_params, target_loss)

    def compute_optimal(self, target_loss: float) -> tuple[float, float]:
        """The ``(N_C, D_C)`` of least compute ``6 N D`` that reaches ``target_loss``.

        At the optimum ``alpha A / N**alpha = beta B / D**beta``, so the two terms
        share the reducible loss ``target_loss - E`` in the ratio ``beta : alpha``.
        """
        gap = target_loss - self.E
        if gap <= 0:
            raise UnreachableLoss(
                f"loss {target_loss} is not above the irreducible loss E = {self.E} "
                f"of the {self.name} law"
            )
        model_term = self.beta * gap / (self.alpha + self.beta)
        data_term = self.alpha * gap / (self.alpha + self.beta)
        return (self.A / model_term) ** (1 / self.alpha), (self.B / data_term) ** (1 / self.beta)


BESIROGLU = ScalingLaw("besiroglu", E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)
CHINCHILLA = ScalingLaw("chinchilla", E=1.6934, A=406.4, B=410.7, alpha=0.3392, beta=0.2849)

#: Predefined laws by name.
LAWS: dict[str, ScalingLaw] = {law.name: law for law in (BESIROGLU, CHINCHILLA)}

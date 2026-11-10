"""Restrictions applied between Calculon's generator and its evaluator.

Calculon's generator (``AllExecutions.all_executions``) yields every
combination of the 13 variables of a parallel plan, about 10**6 for a model at
the scale of Llama 3.1 405B on 16,384 GPUs. MinGPU keeps a subset of them in
cumulative steps (paper, Table 5 and Appendix C), so that the plans are those
Megatron-Core runs and the search stays short:

====  =====================================================  ===================
step  keep an execution if                                   kind
====  =====================================================  ===================
1     no weight, activation or optimizer offload             modelling choice
2     ``t <= t_max`` and ``t`` on network 0 (``t_max`` its    Megatron-Core
      size: the 8 GPUs of an NVLink server)
3     the parallel degrees fit Calculon's network-size rule   lossless
4     the block weights of one GPU fit in its memory          lossless (necessary)
5     tensor-parallel communication ``rs_ag`` (sequence       dominance in data
      parallelism)
6     all-gather redo on wherever Calculon generates it       Megatron-Core
7     no tensor-parallel communication overlap (option)       per run
8     tensor-parallel degree fixed at a given ``t`` (option)  per run
====  =====================================================  ===================

Steps 1 to 6 always apply; 7 and 8 only when asked. Step 3 is Calculon's
``Llm._check_network_assignments`` (``llm.py:1095-1125``) without raising.
Step 4 counts the weights per GPU as Calculon does for multi-head attention
(``llm.py:696-750, 940-980``, ``layers.py:353, 643``), at the size of the
datatype. Step 6: Calculon generates the redo only with ``rs_ag`` and a
recomputation other than ``full`` (``Llm.can_redo_ag``, ``llm.py:252-253``);
elsewhere its only value, off, is kept.

The parallel plans that pass every enabled step are the candidate parallel plans of the pair.
An execution (Calculon's name for a parallel plan) is a tuple in the order of
``Llm.Execution.fields()``, the order in which Calculon's generator yields it.
"""

from __future__ import annotations

from typing import Any

from calculon.llm import Llm
from calculon.system import System

from .gpu_systems import max_tensor_parallel

#: Steps that always apply.
ALWAYS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

EXECUTION_FIELDS: tuple[str, ...] = tuple(Llm.Execution.fields())
_F = {name: i for i, name in enumerate(EXECUTION_FIELDS)}
TP, PP, DP = _F["tensor_par"], _F["pipeline_par"], _F["data_par"]
TN, PN, DN = _F["tensor_par_net"], _F["pipeline_par_net"], _F["data_par_net"]
TPC, REDO, REC = _F["tensor_par_comm_type"], _F["seq_par_ag_redo"], _F["activation_recompute"]
TPO = _F["tensor_par_overlap"]
WO, AO, OO = _F["weight_offload"], _F["activations_offload"], _F["optimizer_offload"]


def _enabled_steps(no_tp_overlap: bool = False, tensor_par: int | None = None) -> list[int]:
    """The steps that apply, in order: 1 to 6, then 7 and 8 when asked."""
    return list(ALWAYS) + ([7] if no_tp_overlap else []) + ([8] if tensor_par is not None else [])


def restriction_settings(
    system: dict[str, Any], no_tp_overlap: bool = False, tensor_par: int | None = None
) -> dict[str, Any]:
    """The restrictions as stored with a search and hashed in the key of a pair.

    Its keys and values must not change: the keys of the cached pairs depend on them.
    """
    return {
        "steps": _enabled_steps(no_tp_overlap, tensor_par),
        "t_max": max_tensor_parallel(system),
        "tensor_par": tensor_par,
    }


class Restrictions:
    """The enabled steps for one model and system, and the test of an execution."""

    def __init__(
        self,
        app: Llm.Application,
        syst: System,
        datatype: str = "bfloat16",
        *,
        no_tp_overlap: bool = False,
        tensor_par: int | None = None,
    ) -> None:
        if datatype not in System.TypeSizes:
            raise ValueError(f"unsupported datatype {datatype!r}")
        self.app = app
        self.nets = [syst.get_network(i) for i in range(syst.num_networks)]
        self.t_max = self.nets[0].size
        self.mem1 = syst.mem1.capacity
        self.weight_bytes = System.TypeSizes[datatype]
        self.tensor_par = tensor_par
        self.steps = _enabled_steps(no_tp_overlap, tensor_par)
        self._weights_fit: dict[tuple[int, int], bool] = {}

    def network_ok(self, tp: int, pp: int, dp: int, tn: int, pn: int, dn: int) -> bool:
        """Calculon's network-size rule (``llm.py:1095-1125``), without raising."""
        used = [False] * len(self.nets)
        size = [1] * len(self.nets)
        for degree, net in ((tp, tn), (pp, pn), (dp, dn)):
            if degree > 1:
                used[net] = True
                size[net] *= degree
        for i, net in enumerate(self.nets):
            if used[i]:
                if size[i] > net.size:
                    return False
                if net.must_be_filled and net.size % size[i] != 0:
                    return False
        return True

    def weights_per_gpu(self, tp: int, pp: int) -> int:
        """Weight elements of the blocks on one GPU, as Calculon counts them."""
        a = self.app
        h, attn = a.hidden, a.attn_heads * a.attn_size
        block = (4 * h * attn + 2 * h * a.feedforward) // tp + 2 * 2 * h
        return block * -(-a.num_blocks // pp)

    def weights_fit(self, tp: int, pp: int) -> bool:
        key = (tp, pp)
        if key not in self._weights_fit:
            self._weights_fit[key] = self.weight_bytes * self.weights_per_gpu(tp, pp) <= self.mem1
        return self._weights_fit[key]

    def passes(self, step: int, e: tuple) -> bool:
        """Whether execution ``e`` passes ``step``."""
        if step == 1:
            return not (e[WO] or e[AO] or e[OO])
        if step == 2:
            return e[TP] <= self.t_max and (e[TP] == 1 or e[TN] == 0)
        if step == 3:
            return self.network_ok(e[TP], e[PP], e[DP], e[TN], e[PN], e[DN])
        if step == 4:
            return self.weights_fit(e[TP], e[PP])
        if step == 5:
            return e[TPC] == "rs_ag"
        if step == 6:
            return e[REDO] or not Llm.can_redo_ag(e[TPC], e[REC])
        if step == 7:
            return e[TPO] == "none"
        if step == 8:
            return e[TP] == self.tensor_par
        raise ValueError(f"unknown step {step}")

    def first_failed(self, e: tuple) -> int | None:
        """The first enabled step that ``e`` fails, or None if it passes them all."""
        for step in self.steps:
            if not self.passes(step, e):
                return step
        return None

    def keep(self, e: tuple) -> bool:
        return self.first_failed(e) is None

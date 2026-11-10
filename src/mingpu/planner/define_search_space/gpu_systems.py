"""GPU systems: the Calculon system file behind each ``--gpu`` name.

``h100``
    ``systems/h100_80g_nvl8_nooffload.json``: Calculon's
    ``systems/h100_80g_nvl8.json`` (80 GiB of HBM, an NVLink domain of 8 GPUs
    and a second network for the rest of the cluster) with two changes: a
    ``bfloat16`` entry under ``matrix`` (the dense peak of 989 TFLOP/s) and
    under ``vector`` (120 TFLOP/s), each with the efficiency curves of the
    ``float16`` entry; and an offload capacity ``mem2.GiB`` of 0 (restriction 1
    of the paper). It is the file of every run of the paper (Appendix C).

A path to a JSON file is accepted wherever a name is. A system is validated by
:class:`SystemConfig` when it is read, since it can be a file of the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class SystemConfig(BaseModel):
    """Top-level shape of a Calculon ``System`` (``calculon/system.py``).

    The nested structures pass through unchanged, since Calculon validates them itself; only
    the top-level keys and the processing mode are checked, for an early, readable error.
    """

    model_config = ConfigDict(extra="forbid")

    matrix: dict[str, Any]
    vector: dict[str, Any]
    mem1: dict[str, Any]
    mem2: dict[str, Any]
    processing_mode: Literal["roofline", "no_overlap"]
    networks: list[dict[str, Any]]


#: System files by GPU name, in the directory ``systems``.
SYSTEMS: dict[str, str] = {"h100": "h100_80g_nvl8_nooffload.json"}


#: ``systems/`` at the root of the repository, the only install MinGPU supports (a clone).
SYSTEMS_DIR = Path(__file__).resolve().parents[4] / "systems"


def load_system(gpu: str) -> dict[str, Any]:
    """The validated Calculon system of ``gpu``: a name of :data:`SYSTEMS` or a JSON path."""
    if gpu in SYSTEMS:
        text = (SYSTEMS_DIR / SYSTEMS[gpu]).read_text()
    elif Path(gpu).is_file():
        text = Path(gpu).read_text()
    else:
        raise ValueError(f"unknown GPU {gpu!r}: use one of {sorted(SYSTEMS)} or a system JSON file")
    return SystemConfig.model_validate(json.loads(text)).model_dump()


def peak_flops(system: dict[str, Any], datatype: str = "bfloat16") -> float:
    """Peak matrix throughput of one device in ``datatype``, in FLOP/s (``F_peak``)."""
    try:
        return system["matrix"][datatype]["tflops"] * 1e12
    except KeyError:
        raise ValueError(f"the system has no matrix entry for {datatype!r}") from None


def max_tensor_parallel(system: dict[str, Any]) -> int:
    """GPUs in one domain of network 0 (``t_max``, 8 for an NVLink server of H100)."""
    return system["networks"][0]["size"]

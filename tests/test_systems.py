import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mingpu.planner.define_search_space.gpu_systems import (
    SYSTEMS,
    load_system,
    max_tensor_parallel,
    peak_flops,
)

CALCULON = Path(__file__).resolve().parents[1] / "vendor" / "calculon"


def test_h100():
    system = load_system("h100")
    assert peak_flops(system) == 989e12
    assert max_tensor_parallel(system) == 8
    assert system["mem2"]["GiB"] == 0
    assert sorted(SYSTEMS) == ["h100"]
    with pytest.raises(ValueError, match="unknown GPU"):
        load_system("b300")
    with pytest.raises(ValueError, match="no matrix entry"):
        peak_flops(system, "float4")


def test_h100_is_calculons_file_with_bfloat16_and_no_offload():
    """The packaged file differs from Calculon's only where its docstring says."""
    ours = load_system("h100")
    calculon = json.loads((CALCULON / "systems" / "h100_80g_nvl8.json").read_text())
    for unit in ("matrix", "vector"):
        added = ours[unit].pop("bfloat16")
        assert added["gflops_efficiency"] == calculon[unit]["float16"]["gflops_efficiency"]
    assert ours["mem2"].pop("GiB") == 0
    calculon["mem2"].pop("GiB")
    assert ours == calculon


def test_system_from_a_file(tmp_path):
    path = tmp_path / "a100.json"
    system = json.loads((CALCULON / "systems" / "a100_80g.json").read_text())
    path.write_text(json.dumps(system))
    assert max_tensor_parallel(load_system(str(path))) == 8
    # a system file read from outside is validated before it is used
    path.write_text(json.dumps({**system, "unexpected": 1}))
    with pytest.raises(ValidationError):
        load_system(str(path))

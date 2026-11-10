import logging

import calculon
import pytest
from calculon.llm import AllExecutions, Llm

from mingpu.planner.define_search_space.candidate_models import Model
from mingpu.planner.define_search_space.gpu_systems import load_system
from mingpu.planner.define_search_space.restrictions import EXECUTION_FIELDS, Restrictions


@pytest.fixture
def setting():
    app = Llm.Application(Model(128).application(8192))
    return app, calculon.System(load_system("h100"))


def execution(**changes):
    """A valid-looking execution of l = 128 on 64 GPUs, with some fields changed."""
    base = dict(
        num_procs=64, tensor_par=8, pipeline_par=8, data_par=1, tensor_par_net=0,
        pipeline_par_net=1, data_par_net=0, batch_size=16, microbatch_size=1,
        datatype="bfloat16", fused_activation=True, attention_type="multihead",
        activation_recompute="full", pipeline_interleaving=1, optimizer_sharding=False,
        tensor_par_comm_type="rs_ag", tensor_par_overlap="none", seq_par_ag_redo=False,
        data_par_overlap=False, weight_offload=False, activations_offload=False,
        optimizer_offload=False, training=True,
    )  # fmt: skip
    base.update(changes)
    return tuple(base[name] for name in EXECUTION_FIELDS)


def test_enabled_steps(setting):
    assert Restrictions(*setting).steps == [1, 2, 3, 4, 5, 6]
    assert Restrictions(*setting, no_tp_overlap=True).steps == [1, 2, 3, 4, 5, 6, 7]
    assert Restrictions(*setting, tensor_par=8).steps == [1, 2, 3, 4, 5, 6, 8]


def test_each_step(setting):
    r = Restrictions(*setting, no_tp_overlap=True, tensor_par=8)
    assert r.keep(execution())
    assert r.first_failed(execution(optimizer_offload=True)) == 1
    assert r.first_failed(execution(tensor_par_net=1)) == 2
    assert r.first_failed(execution(tensor_par=16, pipeline_par=4)) == 2
    assert r.first_failed(execution(pipeline_par_net=0)) == 3  # 8 x 8 on an 8-GPU network
    assert r.first_failed(execution(pipeline_par=1, data_par=8, data_par_net=1)) == 4
    assert r.first_failed(execution(tensor_par_comm_type="ar")) == 5
    assert r.first_failed(execution(activation_recompute="none")) == 6  # redo off where allowed
    assert r.keep(execution(activation_recompute="none", seq_par_ag_redo=True))
    assert r.first_failed(execution(tensor_par_overlap="ring")) == 7
    assert Restrictions(*setting).keep(execution(tensor_par_overlap="ring"))
    assert r.first_failed(execution(tensor_par=4, pipeline_par=16, data_par=1)) == 8


def test_weights_follow_the_datatype(setting):
    bf16 = Restrictions(*setting, "bfloat16")
    fp32 = Restrictions(*setting, "float32")
    assert fp32.weights_per_gpu(8, 2) == bf16.weights_per_gpu(8, 2)
    assert bf16.weights_fit(8, 2) and not fp32.weights_fit(8, 2)  # 51.5 GB, 103 GB
    with pytest.raises(ValueError):
        Restrictions(*setting, "int4")


def test_t_max_is_the_size_of_network_0(setting):
    app, _ = setting
    system = load_system("h100")
    system["networks"][0]["size"] = 4
    r = Restrictions(app, calculon.System(system))
    assert r.t_max == 4
    assert r.first_failed(execution(num_procs=32, data_par=1, pipeline_par=4)) == 2


def test_step_3_is_calculons_network_rule():
    """Step 3 agrees with Calculon's own network check on every generated degree and network."""
    app = Llm.Application(Model(8).application(2048))
    syst = calculon.System(load_system("h100"))
    r = Restrictions(app, syst)
    fields = [EXECUTION_FIELDS.index(f) for f in ("tensor_par", "pipeline_par", "data_par")]
    nets = [
        EXECUTION_FIELDS.index(f) for f in ("tensor_par_net", "pipeline_par_net", "data_par_net")
    ]
    seen = {}
    for e in AllExecutions.all_executions(app, syst, 16, 32, "bfloat16", [True]):
        key = tuple(e[i] for i in fields + nets)
        if key in seen:
            continue
        model = Llm(app, logging.getLogger("test"))
        model.exe, model.sys = Llm.Execution(*e), syst
        try:
            model._check_network_assignments()
            calculon_ok = True
        except Llm.Error:
            calculon_ok = False
        assert r.network_ok(*key) == calculon_ok, key
        seen[key] = calculon_ok
    assert True in seen.values() and False in seen.values()

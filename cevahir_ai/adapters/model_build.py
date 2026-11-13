"""Architecture facts from the legacy builder: the unchanged normaliser and the constructor.

``normalize`` wraps ``model_management.config_schema.normalize_model_config`` with the profile the
builder and the load checks use, so the architecture key describes what the builder builds.
``expected_layout`` reads the weight layout from the constructor on the meta device.
"""
import contextlib
import importlib.util
import logging
import sys

from cevahir_ai._compat.repo import import_legacy, repo_root
from cevahir_ai.keys.schemes.arch import arch_cfg_key

_PRIVATE_SCHEMA = "cevahir_ai._legacy_config_schema"


def config_schema():
    """The legacy config schema module.

    ``model_management/__init__`` imports torch eagerly, while ``config_schema.py`` itself needs only
    the standard library. When ``model_management`` is not loaded yet the file is executed under a
    private module name, so architecture keys stay torch-free; the legacy package is not shadowed.
    """
    if "model_management" in sys.modules:
        return import_legacy("model_management.config_schema")
    if _PRIVATE_SCHEMA not in sys.modules:
        path = repo_root() / "model_management" / "config_schema.py"
        spec = importlib.util.spec_from_file_location(_PRIVATE_SCHEMA, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_PRIVATE_SCHEMA] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[_PRIVATE_SCHEMA]
            raise
    return sys.modules[_PRIVATE_SCHEMA]


def normalize(config):
    """``normalize_model_config(config, legacy_profile="model_manager")``, called unchanged."""
    return config_schema().normalize_model_config(config, legacy_profile="model_manager")


def arch_of_config(config):
    """Returns ``(arch:cfg-v1 key, function-defining fields, normalised config)`` of a raw config."""
    normalized = normalize(config)
    key, fields = arch_cfg_key(normalized)
    return key, fields, normalized


def arch_of_model(model):
    """Architecture of a live model, from its construction truth (``checkpoint_contract.model_config``)."""
    stored = import_legacy("model_management.checkpoint_contract").model_config(model)
    if not stored:
        return None, None, None
    return arch_of_config(stored)


@contextlib.contextmanager
def _quiet():
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous)


def build_meta(config):
    """Build ``CevahirNeuralNetwork`` for ``config`` on the meta device: structure, no memory, no data.

    The constructor arguments are chosen exactly as ``ModelInitializer.build_model`` chooses them.
    The global RNG state is restored afterwards, so a readout never shifts anybody's seed.
    """
    import torch
    network = import_legacy("src.neural_network").CevahirNeuralNetwork
    initializer = import_legacy("model_management.model_initializer")
    normalized = normalize(config)
    kwargs = initializer._filter_kwargs_for_ctor(network, normalized)
    kwargs.update(device="cpu", use_tensorboard=False, tb_writer=None)
    rng = torch.get_rng_state()
    try:
        with _quiet(), torch.device("meta"):
            return network(**kwargs)
    finally:
        torch.set_rng_state(rng)


def structure_model(config):
    """A model that has the structure of ``config`` as cheaply as possible. Returns ``(model, how)``.

    1. meta device (no memory);
    2. YaRN/linear RoPE scaling computes its frequency table from tensor *values*, which meta tensors
       do not have: the scaling is neutralised and the build repeated on meta. RoPE tables are
       non-persistent buffers, so the state-dict layout is the same (a test pins this against a
       real build);
    3. a real CPU build, whatever it costs, with the RNG state restored.
    """
    import torch
    try:
        return build_meta(config), "meta"
    except NotImplementedError:
        pass
    normalized = normalize(config)
    if normalized.get("pe_mode") == "rope" and normalized.get("rope_scaling_type") not in (None, "none"):
        try:
            return build_meta({**normalized, "rope_scaling_type": "none"}), "meta (RoPE scaling neutralised)"
        except NotImplementedError:
            pass
    rng = torch.get_rng_state()
    try:
        return build_from_declared(normalized), "cpu"
    finally:
        torch.set_rng_state(rng)


def expected_layout(config):
    """Names, shapes, dtypes, alias groups and non-persistent buffers the current code builds for ``config``."""
    from cevahir_ai.keys.torchkeys.wschema import layout_of_module
    model, how = structure_model(config)
    layout = layout_of_module(model)
    layout["built_on"] = how
    return layout


def build_model(config):
    """Build a real model through the legacy path CEV-11 uses: ``ModelManager(config).initialize(...)``."""
    manager = import_legacy("model_management.model_manager").ModelManager(dict(config))
    manager.initialize(build_optimizer=False, build_criterion=False, build_scheduler=False)
    return manager.model


def build_from_declared(config):
    """Build the model a checkpoint declares, on CPU, from its architecture fields only.

    A stored config can carry runtime settings of the run that wrote it (TensorBoard directories,
    compile flags, a CUDA device). Only the 47 normalised architecture keys are passed on, so
    rebuilding a model for inspection or evaluation has no side effects.
    """
    from cevahir_ai.keys.classify import ARCH_FIELD_CLASS_V1
    normalized = normalize(config)
    clean = {name: value for name, value in normalized.items() if name in ARCH_FIELD_CLASS_V1}
    clean.update(device="cpu", log_level=50)
    with _quiet():
        return build_model(clean)

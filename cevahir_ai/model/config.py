"""Configuration schema facade. ``SystemConfig`` is ``model_management.config_schema.CevahirConfig``
(the serving facade's class of the same legacy name is ``cevahir_ai.serving.FacadeConfig``)."""
from cevahir_ai._lazy import lazy_exports
from cevahir_ai._roof import EXPORTS

__all__ = sorted(EXPORTS["model.config"])
__getattr__, __dir__ = lazy_exports(__name__, EXPORTS["model.config"])

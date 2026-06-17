from __future__ import annotations

from importlib import import_module

__all__ = ["DEFAULT_CONFIG", "run_paper_reproduction"]


def __getattr__(name: str):
    if name in __all__:
        experiment = import_module("repro.experiment")
        return getattr(experiment, name)
    raise AttributeError(name)

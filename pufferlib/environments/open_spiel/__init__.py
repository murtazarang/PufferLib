"""Load environment factories independently of optional policies."""

from importlib import import_module

from .environment import env_creator, make


def __getattr__(name):
    """Import an optional policy only when that policy is requested."""
    if name in {"Policy", "Recurrent"}:
        return getattr(import_module(f"{__name__}.torch"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

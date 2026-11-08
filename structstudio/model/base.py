"""Shared helpers for model objects.

Model objects are plain dataclasses.  Numeric fields carry a ``quantity``
entry in their field metadata so the model can be converted between unit
systems and so the mesher knows how to convert a value to SI.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any

from ..units import UnitSystem


def q(quantity: str, default: Any = 0.0, **kw):
    """Declare a numeric dataclass field with a physical quantity."""
    kw.setdefault("metadata", {})
    kw["metadata"] = {**kw["metadata"], "quantity": quantity}
    if callable(default) and not isinstance(default, (int, float)):
        return field(default_factory=default, **kw)
    return field(default=default, **kw)


def qlist(quantity: str, default_factory, **kw):
    kw.setdefault("metadata", {})
    kw["metadata"] = {**kw["metadata"], "quantity": quantity}
    return field(default_factory=default_factory, **kw)


class ModelError(Exception):
    """A structured model/analysis error with a machine readable code."""

    def __init__(self, code: str, message: str, path: str | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path

    def to_dict(self) -> dict:
        d = {"code": self.code, "message": self.message}
        if self.path:
            d["path"] = self.path
        return d


def convert_dataclass(obj, src: UnitSystem, dst: UnitSystem):
    """Return a copy of ``obj`` with all quantity-tagged fields converted."""
    if src == dst:
        return obj
    hook = getattr(obj, "convert_units", None)
    if callable(hook):
        return hook(src, dst)
    return convert_tagged(obj, src, dst)


def convert_tagged(obj, src: UnitSystem, dst: UnitSystem):
    """The quantity-tagged fields (and nested dataclasses) of ``obj`` converted;
    the default of :func:`convert_dataclass`, also for ``convert_units`` hooks."""
    if src == dst:
        return obj
    changes = {}
    for f in fields(obj):
        quantity = f.metadata.get("quantity")
        value = getattr(obj, f.name)
        if quantity is None:
            if dataclasses.is_dataclass(value):
                changes[f.name] = convert_dataclass(value, src, dst)
            elif isinstance(value, list) and value and dataclasses.is_dataclass(value[0]):
                changes[f.name] = [convert_dataclass(v, src, dst) for v in value]
            elif isinstance(value, dict) and value and all(dataclasses.is_dataclass(v) for v in value.values()):
                changes[f.name] = {k: convert_dataclass(v, src, dst) for k, v in value.items()}
            continue
        if value is None:
            continue
        ratio = src.factor(quantity) / dst.factor(quantity)
        if isinstance(value, (list, tuple)):
            changes[f.name] = type(value)(None if v is None else v * ratio for v in value)
        elif isinstance(value, dict):
            changes[f.name] = {k: v * ratio for k, v in value.items()}
        else:
            changes[f.name] = value * ratio
    return dataclasses.replace(obj, **changes)


def to_dict(obj) -> Any:
    """Recursively convert dataclasses to JSON compatible dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {"_type": type(obj).__name__}
        for f in fields(obj):
            d[f.name] = to_dict(getattr(obj, f.name))
        return d
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


__all__ = ["q", "qlist", "ModelError", "convert_dataclass", "convert_tagged", "to_dict", "dataclass", "field"]

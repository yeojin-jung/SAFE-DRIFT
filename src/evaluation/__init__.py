from __future__ import annotations


def get_evaluator(name: str):
    from .registry import get_evaluator as _get_evaluator

    return _get_evaluator(name)

__all__ = ["get_evaluator"]

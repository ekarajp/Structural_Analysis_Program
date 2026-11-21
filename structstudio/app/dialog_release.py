"""Forms run with ``exec()`` are released when they are done with.

A form made with the window as its parent (``d = Form(self); d.exec()``) belongs to the
window: Qt keeps it - with all its widgets and the copies of the model data it holds -
until the program closes.  Every opening of Define > Material Properties kept about
1.8 MB; a long session of opening forms grew by hundreds of megabytes.

:func:`install` makes ``QDialog.exec`` hand each form to :func:`_release` once the
command that opened it has finished: a form that nothing in Python holds any more (no
attribute, no variable of a command still running) and that is not shown is deleted.
A form somebody still holds is left alone - exactly as before.

"Nothing holds it" is counted: the references to the form must be no more than the one
of its parent (if the parent still has one) plus the form's own: its callbacks
(``lambda: self...`` and ``self.method`` connected to the signals of its own widgets)
and its own widgets / helper objects that point back at it.  A callback
that something else keeps (a list of listeners, an attribute of another object) counts
as a holder, and so does every variable or attribute that holds the form.
"""
from __future__ import annotations

import gc
import sys
import types

from PySide6 import QtCore, QtWidgets

_state = {"installed": False}
_pending: dict = {}  # id -> the form, between its exec() and its release
RETRIES = 6  # a command still running (a second form on top) holds the form a while longer
RETRY_MS = 1500


def _own_callbacks(d, first) -> int:
    """How many of the references to ``d`` (``first``: its referrers) are its own
    callbacks: closures over it and bound methods of it that only signal connections
    keep (-1: one of them is kept by something else - the form is in use)."""
    n = 0
    mine = {id(first)}  # the lists made here hold what they list: they are no holders

    def holders(obj):
        # (plain loops: a comprehension over a generator would keep each object in a
        # cell of its own, and that cell would be found as a holder)
        got = gc.get_referrers(obj)
        mine.add(id(got))
        out = []
        for h in got:
            if isinstance(h, types.FrameType) or id(h) in mine:
                continue
            out.append(h)
        mine.add(id(out))
        return out

    def descends(obj) -> bool:  # a widget / object inside the form
        try:
            p = obj
            for _ in range(40):
                p = p.parent()
                if p is None:
                    return False
                if p is d:
                    return True
        except Exception:
            pass
        return False

    def part_of_form(obj) -> bool:
        """An object of the form itself: one of its widgets, or a helper only the form
        (or one of its widgets) holds."""
        if isinstance(obj, QtCore.QObject):
            return obj is not d and descends(obj)
        for h in holders(obj):
            if h is d:
                continue
            if isinstance(h, dict):  # the attributes of ... whom?
                who = holders(h)
                if len(who) == 1 and (who[0] is d or (isinstance(who[0], QtCore.QObject) and descends(who[0]))):
                    continue
            return False
        return True

    for r in first:
        if isinstance(r, dict):  # the attributes of one of the form's own widgets / helpers
            who = holders(r)
            if len(who) == 1 and who[0] is not d and part_of_form(who[0]):
                n += 1
            continue
        if isinstance(r, QtCore.QObject):
            if descends(r):
                n += 1
            continue
        if isinstance(r, types.MethodType) and r.__self__ is d:
            if holders(r):
                return -1
            n += 1
        elif isinstance(r, types.CellType):
            n += 1
            for tup in holders(r):  # the closure tuples of the functions made over it
                if not isinstance(tup, tuple):
                    return -1
                for fn in holders(tup):
                    if not isinstance(fn, types.FunctionType):
                        return -1
                    for h in holders(fn):
                        if h is not tup:
                            return -1  # in a list, a dict, another closure ...: somebody else calls it
        elif not isinstance(r, (types.FrameType, type)) and part_of_form(r):
            n += 1  # a helper object of the form that points back at it
    return n


def _release(key: int, tries: int = 0) -> None:
    d = _pending.pop(key, None)  # (our own reference: not counted below)
    try:
        if d is None:
            return
        import shiboken6
        if not shiboken6.isValid(d) or d.isVisible():
            return
        parent = d.parent()
        if parent is None:
            return  # no parent: Python frees it with its last reference
        probe = object()
        mine = sys.getrefcount(probe)  # what this function itself adds to a count (a name, the argument)
        n = sys.getrefcount(d)
        first = gc.get_referrers(d)
        owner = 0
        for r in first:  # the parent's own reference to its child (dropped once exec() has run)
            if r is parent:
                owner = 1
        own = _own_callbacks(d, first)
        del first
        if own >= 0 and n <= mine + owner + own:
            d.deleteLater()
        elif tries < RETRIES:
            _pending[key] = d
            QtCore.QTimer.singleShot(RETRY_MS, lambda: _release(key, tries + 1))
    except Exception:  # (releasing memory must never break a command)
        pass


def install() -> None:
    """Idempotent; called when the main window is made."""
    if _state["installed"]:
        return
    original = QtWidgets.QDialog.exec

    def exec_and_release(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            try:
                key = id(self)
                _pending[key] = self
                QtCore.QTimer.singleShot(0, lambda: _release(key))
            except Exception:
                pass

    QtWidgets.QDialog.exec = exec_and_release
    _state["installed"] = True

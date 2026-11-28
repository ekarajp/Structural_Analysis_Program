"""CAD drawings placed in the model as *references* to draw over.

A reference is no part of the structure: it adds nothing to the analysis, changing one
keeps the analysis results and the model stays locked (as design data).  It names a
drawing file - linked (read from disk, reloaded when the file changes), embedded (a copy
kept in the model file), or both - and says where the drawing lies:

* ``view`` ``"plan"``: on the plans of ``stories`` (or at elevation ``level`` when it is
  not tied to a story); ``"xz"`` / ``"yz"``: on the elevation at ordinate ``level``
  (y of an X-Z elevation, x of a Y-Z elevation), drawing x along the elevation and
  drawing y up;
* file point ``base`` (file units) is put at ``origin`` (the two coordinates of the view
  plane, model length units), turned by ``rotation`` and scaled by the drawing ``unit``
  and by ``scale`` (a drawing that was not drawn 1:1).

A drawing is a DXF file or one ``page`` of a PDF plotted from CAD.  A PDF page is read in
millimetres on the sheet (``unit`` "mm"); ``scale`` is then its plot scale (100 for a
sheet plotted 1:100).

The embedded copies are :class:`Blob` objects in ``Model.attachments``.  A blob is
immutable and is pickled *by its key* (the undo history keeps keys, not megabytes).
"""
from __future__ import annotations

import base64
import hashlib
import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

UNIT_TO_M = {"mm": 0.001, "cm": 0.01, "dm": 0.1, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048,
             "yd": 0.9144, "mi": 1609.344}
REFERENCE_VIEWS = ("plan", "xz", "yz")

_BLOBS: Dict[str, "Blob"] = {}  # every blob of this session by key (see Blob.__reduce__)


class Blob:
    """The bytes of an embedded file (kept compressed).  ``key``: the SHA-1 of the file."""

    __slots__ = ("key", "name", "size", "_z", "_b64", "__weakref__")

    def __init__(self, data: bytes, name: str = ""):
        data = bytes(data)
        self.key = hashlib.sha1(data).hexdigest()
        self.name = name
        self.size = len(data)
        self._z = zlib.compress(data, 6)
        self._b64: Optional[str] = None
        _BLOBS.setdefault(self.key, self)

    @property
    def data(self) -> bytes:
        return zlib.decompress(self._z) if self._z else b""

    @property
    def stored_size(self) -> int:
        return len(self._z)

    def to_dict(self) -> dict:
        if self._b64 is None:
            self._b64 = base64.b64encode(self._z).decode("ascii")
        return {"name": self.name, "size": self.size, "zlib": self._b64}

    @classmethod
    def from_dict(cls, key: str, d: dict) -> "Blob":
        known = _BLOBS.get(key)
        if known is not None:
            return known
        self = cls.__new__(cls)
        self.key = str(key)
        self.name = str(d.get("name", ""))
        self.size = int(d.get("size", 0))
        self._b64 = d.get("zlib") or ""
        self._z = base64.b64decode(self._b64) if self._b64 else b""
        _BLOBS[self.key] = self
        return self

    def __reduce__(self):
        return (_blob_by_key, (self.key, self.name, self.size))

    def __deepcopy__(self, memo):
        return self

    def __eq__(self, other):
        return isinstance(other, Blob) and other.key == self.key

    def __hash__(self):
        return hash(self.key)

    def __repr__(self):
        return f"<Blob {self.name!r} {self.size} bytes {self.key[:8]}>"


def _blob_by_key(key: str, name: str = "", size: int = 0) -> Blob:
    b = _BLOBS.get(key)
    if b is None:  # (another process: the model without its attachments' bytes)
        b = Blob.__new__(Blob)
        b.key, b.name, b.size, b._z, b._b64 = key, name, size, b"", None
    return b


@dataclass
class CadReference:
    """One drawing placed as a reference (see the module text)."""

    name: str
    path: str = ""  # the linked file as it was chosen ("": embedded only)
    rel_path: str = ""  # the same, relative to the model file (written when the model is saved)
    embedded: Optional[str] = None  # key of the copy in Model.attachments (None: linked only)
    view: str = "plan"
    stories: List[str] = field(default_factory=list)
    level: float = 0.0  # length (model units)
    unit: str = "m"  # the drawing unit of the file
    scale: float = 1.0
    base: Tuple[float, float] = (0.0, 0.0)  # file units
    origin: Tuple[float, float] = (0.0, 0.0)  # length (model units)
    rotation: float = 0.0  # degrees, counter-clockwise in the view plane
    visible: bool = True
    pinned: bool = True
    snap: bool = True
    on_top: bool = True  # drawn over the model (False: under it)
    color: str = "gray"  # "gray" | "layers" (the drawing's own colours) | "#rrggbb"
    strength: float = 0.6  # 0..1: how strongly it is drawn (1 = full colour)
    layers_off: List[str] = field(default_factory=list)
    page: int = 1  # the page of a PDF drawing (1 = the first)
    texts: bool = True  # its texts are drawn (never snapped to)

    def convert_units(self, src, dst) -> "CadReference":
        import dataclasses
        r = src.factor("length") / dst.factor("length")
        return dataclasses.replace(self, level=self.level * r, origin=tuple(v * r for v in self.origin),
                                   stories=list(self.stories), layers_off=list(self.layers_off),
                                   base=tuple(self.base))

    # ------------------------------------------------------------- where it lies
    def factor(self, length_unit: str) -> float:
        """Model length per drawing unit (with the reference's own scale)."""
        return UNIT_TO_M.get(self.unit, 1.0) / UNIT_TO_M.get(length_unit, 1.0) * float(self.scale)

    def matrix(self, length_unit: str) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """The 2 x 3 matrix taking file (x, y) to the two coordinates of the view plane."""
        s = self.factor(length_unit)
        a = math.radians(self.rotation)
        c, sn = math.cos(a) * s, math.sin(a) * s
        bx, by = self.base
        ox, oy = self.origin
        return ((c, -sn, ox - (c * bx - sn * by)), (sn, c, oy - (sn * bx + c * by)))

    def to_plane(self, xy, length_unit: str) -> Tuple[float, float]:
        (a, b, tx), (c, d, ty) = self.matrix(length_unit)
        return (a * xy[0] + b * xy[1] + tx, c * xy[0] + d * xy[1] + ty)

    def to_file(self, uv, length_unit: str) -> Tuple[float, float]:
        s = self.factor(length_unit)
        a = math.radians(self.rotation)
        c, sn = math.cos(a), math.sin(a)
        du, dv = (uv[0] - self.origin[0]) / s, (uv[1] - self.origin[1]) / s
        return (self.base[0] + c * du + sn * dv, self.base[1] - sn * du + c * dv)

    def rebase(self, uv, length_unit: str) -> None:
        """Keep the picture where it is and make plane point ``uv`` the base / origin pair
        (the fixed point of a later turn or scale)."""
        self.base = self.to_file(uv, length_unit)
        self.origin = (float(uv[0]), float(uv[1]))

    def shown_on(self, view: str, value: float, story: Optional[str], tol: float, known_stories=()) -> bool:
        """Is it drawn in a window of this ``view`` at plane ``value`` (plan: of ``story``)?"""
        if view != self.view or not self.visible:
            return False
        if view == "plan":
            mine = [s for s in self.stories if not known_stories or s in known_stories]
            if mine:
                return story in mine
        return abs(float(value) - float(self.level)) <= tol

    def source_label(self) -> str:
        import os
        return os.path.basename(self.path.replace("\\", "/")) or self.name


def align_two_points(ref: CadReference, length_unit: str, p1, q1, p2, q2, scale: bool = False) -> float:
    """Move, turn (and with ``scale`` resize) ``ref`` so that the plane points p1, p2 of
    its picture come to q1, q2: p1 lands exactly on q1, p2 on the ray q1 -> q2 (exactly
    on q2 with ``scale``).  Returns the ratio |q1 q2| / |p1 p2| (1 = the picture has the
    size of the model)."""
    d0 = (p2[0] - p1[0], p2[1] - p1[1])
    d1 = (q2[0] - q1[0], q2[1] - q1[1])
    L0, L1 = math.hypot(*d0), math.hypot(*d1)
    if L0 <= 0 or L1 <= 0:
        raise ValueError("the two points of a pair are the same point")
    ref.rebase(p1, length_unit)
    ref.origin = (float(q1[0]), float(q1[1]))
    ref.rotation = (ref.rotation + math.degrees(math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0]))) % 360.0
    ratio = L1 / L0
    if scale:
        ref.scale = float(ref.scale) * ratio
    return ratio

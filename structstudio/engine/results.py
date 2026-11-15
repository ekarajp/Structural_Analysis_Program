"""Raw (SI) result containers produced by the solvers."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Dict, List, Optional

import numpy as np


@dataclass
class ModalData:
    periods: np.ndarray  # (n_modes,)
    omega: np.ndarray
    eigenvalues: np.ndarray
    shapes: np.ndarray  # (n_full, n_modes) mass-normalised, full DOF set
    participation: np.ndarray  # (n_modes, 6)  UX UY UZ RX RY RZ (rotations about centre of mass)
    mass_ratios: np.ndarray  # (n_modes, 6)
    total_mass: np.ndarray  # (6,) unrestrained mass / rotational inertia per direction
    center_of_mass: np.ndarray  # (3,)
    method: str = "Eigen"  # the method actually used (effective)
    requested_method: str = "Eigen"  # the case's method (Ritz is not implemented: Eigen used)
    solver: Optional[dict] = None  # requested / effective shift, tolerance, residuals, mass coverage

    @property
    def n_modes(self) -> int:
        return len(self.periods)

    @property
    def frequencies(self) -> np.ndarray:
        return 1.0 / self.periods


_GAUSS = None


class ShellResultants(Mapping):
    """Element id -> resultants dict of every shell of a case, kept as a few arrays.

    The same as the dict of dicts it replaces (``r[eid]["M_nodes"]`` ...), but one
    case of a large model takes a third of the memory: the dicts and the small
    arrays of every element are made only when an element is asked for."""

    KEYS = ("F", "M", "V", "F_nodes", "M_nodes", "V_nodes")

    def __init__(self, ids, F, M, V, Fn, Mn, Vn, gauss=None):
        self.ids = np.asarray(ids, dtype=np.int64)
        self.F, self.M, self.V, self.Fn, self.Mn, self.Vn = F, M, V, Fn, Mn, Vn
        self.gauss = gauss
        self._pos = None

    def _index(self) -> dict:
        if self._pos is None:
            self._pos = {int(e): k for k, e in enumerate(self.ids.tolist())}
        return self._pos

    def position(self, eids) -> np.ndarray:
        """Rows of the arrays for the element ids ``eids``."""
        pos = self._index()
        return np.array([pos[int(e)] for e in eids], dtype=np.int64)

    def __getitem__(self, eid) -> dict:
        k = self._index()[int(eid)]
        return {"gauss": self.gauss, "F": self.F[k], "M": self.M[k], "V": self.V[k],
                "F_nodes": self.Fn[k], "M_nodes": self.Mn[k], "V_nodes": self.Vn[k]}

    def __iter__(self):
        return iter(self.ids.tolist())

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, eid) -> bool:
        try:
            return int(eid) in self._index()
        except (TypeError, ValueError):
            return False

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_pos"] = None
        return d


@dataclass
class CaseResult:
    """Results of one load case over the analysis model (SI).

    For envelope type cases (response spectrum) the arrays hold the positive
    maxima; the minima are their negatives.  ``modes`` holds the per-mode
    signed responses used for combinations/diagnostics.
    """

    case: str
    type: str
    displacements: np.ndarray  # (n_nodes, 6) full DOF set, global
    reactions: Dict[int, np.ndarray]  # node id -> (6,) global reaction (restrained + spring)
    frame_end_forces: np.ndarray  # (n_frames, 12) local: forces nodes apply to element at I and J
    frame_loads: Dict[int, list] = field(default_factory=dict)  # element id -> ElementLoads (for station forces)
    # element id -> resultants dict (a ShellResultants: arrays, dict-like)
    shell_resultants: Mapping = field(default_factory=dict)
    is_envelope: bool = False
    modal: Optional[ModalData] = None
    mode_number: Optional[int] = None
    modal_displacements: Optional[np.ndarray] = None  # (n_modes, n_nodes, 6) signed, for RS
    modal_frame_forces: Optional[np.ndarray] = None  # (n_modes, n_frames, 12)
    modal_reactions: Optional[Dict[int, np.ndarray]] = None  # node -> (n_modes, 6)
    spectrum_base: Optional[np.ndarray] = None  # (6,) SI, about the origin; set for spectrum cases
    diagnostics: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    # total (global, SI) reaction of distributed supports (frame elastic foundations)
    distributed_reaction: Optional[np.ndarray] = None
    # nonlinear / time history cases: step or time histories (see engine.nonlinear.runner)
    history: Optional[dict] = None

    def base_reaction(self) -> np.ndarray:
        """Sum of reactions (forces only, moments not transferred)."""
        tot = np.zeros(6)
        for r in self.reactions.values():
            tot[:3] += r[:3]
        if self.distributed_reaction is not None:
            tot[:3] += self.distributed_reaction[:3]
        return tot

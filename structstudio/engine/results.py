"""Raw (SI) result containers produced by the solvers."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    method: str = "Eigen"

    @property
    def n_modes(self) -> int:
        return len(self.periods)

    @property
    def frequencies(self) -> np.ndarray:
        return 1.0 / self.periods


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
    shell_resultants: Dict[int, dict] = field(default_factory=dict)  # element id -> resultants dict
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

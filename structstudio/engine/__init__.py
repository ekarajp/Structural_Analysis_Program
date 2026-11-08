"""Finite element engine (SI): elements, assembly, kernel, solvers."""
from .assembly import Assembler
from .kernel import Kernel, SciPyKernel, default_kernel
from .results import CaseResult, ModalData
from .solvers import Engine

__all__ = ["Assembler", "Kernel", "SciPyKernel", "default_kernel", "CaseResult", "ModalData", "Engine"]

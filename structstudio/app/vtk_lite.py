"""The VTK the program uses, without loading all of VTK.

``import vtk`` loads every VTK module (well over a hundred libraries): about 1.6 s
here, and on Windows - where each library is also scanned by the antivirus the
first time - many seconds of the program's start.  The program uses some fifty
VTK classes from the modules below; this module loads only those, plus the
rendering back ends VTK picks its implementations from (OpenGL, text, the
interactor styles).  Use it as ``from . import vtk_lite as vtk``.

A name not found here loads the whole of VTK once and is taken from there, so a
class added to the program later still works (only slower to start).
"""
import os as _os

if _os.environ.get("STRUCTSTUDIO_VTK_FULL"):  # safety switch: the whole of VTK, as before
    from vtkmodules.all import *  # noqa: F401,F403
from vtkmodules.vtkCommonCore import *  # noqa: F401,F403
from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR  # noqa: F401
from vtkmodules.vtkCommonDataModel import *  # noqa: F401,F403
from vtkmodules.vtkCommonMath import *  # noqa: F401,F403
from vtkmodules.vtkCommonTransforms import *  # noqa: F401,F403
from vtkmodules.vtkFiltersCore import *  # noqa: F401,F403
from vtkmodules.vtkFiltersGeneral import *  # noqa: F401,F403
from vtkmodules.vtkFiltersSources import *  # noqa: F401,F403
from vtkmodules.vtkIOImage import *  # noqa: F401,F403
from vtkmodules.vtkInteractionStyle import *  # noqa: F401,F403
from vtkmodules.vtkInteractionWidgets import *  # noqa: F401,F403
from vtkmodules.vtkRenderingAnnotation import *  # noqa: F401,F403
from vtkmodules.vtkRenderingCore import *  # noqa: F401,F403
from vtkmodules.vtkRenderingFreeType import *  # noqa: F401,F403
from vtkmodules.vtkRenderingLabel import *  # noqa: F401,F403
# the back ends: the OpenGL implementations of the rendering classes, the window /
# interactor of the platform, the 2-D context (labels, scalar bar)
import vtkmodules.vtkRenderingOpenGL2  # noqa: F401,E402
import vtkmodules.vtkRenderingUI  # noqa: F401,E402
try:
    import vtkmodules.vtkRenderingContextOpenGL2  # noqa: F401
except ImportError:  # (not in every VTK build)
    pass
from vtkmodules import util  # noqa: F401,E402  (vtk.util.numpy_support)
import vtkmodules.util.numpy_support  # noqa: F401,E402


def __getattr__(name):
    """Anything else: from the whole of VTK (loaded once, on first use)."""
    import vtkmodules.all as _all
    try:
        return getattr(_all, name)
    except AttributeError:
        raise AttributeError(f"module 'vtk' has no attribute {name!r}") from None

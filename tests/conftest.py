"""Test session setup.

The desktop tests set QT_QPA_PLATFORM=offscreen when nothing is set (Linux CI: no
display, xvfb for the VTK windows).  On Windows the offscreen platform gives VTK no
OpenGL context: creating a 3-D view there ends in an access violation.  A Windows
session always has a desktop, so the native platform is used (the test windows open
and close by themselves)."""
import os
import sys

if sys.platform == "win32" and not os.environ.get("QT_QPA_PLATFORM"):
    os.environ["QT_QPA_PLATFORM"] = "windows"

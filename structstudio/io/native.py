"""Native CSI model files (.edb / .sdb / .fdb / .bdb).

The binary formats are proprietary and change with every release, so they are
not parsed here.  Instead the CSI program installed on the computer opens the
file through its official API and writes it as text (``csi_convert.ps1``):
all model-definition database tables, and for ETABS also the ``.e2k`` text
model.  Windows only; needs a licensed ETABS / SAP2000 / SAFE / CSiBridge.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from typing import List, Optional

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csi_convert.ps1")


class ConversionError(RuntimeError):
    pass


def default_out_dir(path: str) -> str:
    """Conversion folder of one input file: its name plus a key of its full path and type,
    so that two models with the same name (another folder, another program) never share
    a folder and read each other's converted files."""
    import hashlib
    base, ext = os.path.splitext(os.path.basename(path))
    key = hashlib.sha1(os.path.normcase(os.path.abspath(path)).encode("utf-8")).hexdigest()[:10]
    root = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".structstudio")
    return os.path.join(root, "StructuralStudio", "import", f"{base}{ext.replace('.', '-')}-{key}")


def installed_programs() -> List[str]:
    """Names of the CSI program folders found on this computer."""
    out = []
    for root in {os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", "")}:
        if root:
            out += [os.path.basename(p) for p in glob.glob(os.path.join(root, "Computers and Structures", "*"))]
    return sorted(set(out))


def convert(path: str, out_dir: Optional[str] = None, timeout: float = 900.0) -> List[str]:
    """Convert a native file with the installed program; returns the files written."""
    if sys.platform != "win32":
        raise ConversionError(
            f"{os.path.basename(path)} is a native CSI model file.  It can only be converted on Windows with the "
            "program installed.  Alternatively export it from the program as text (ETABS: File > Export > "
            "ETABS .e2k; SAP2000: File > Export > SAP2000 .s2k; SAFE: File > Export > SAFE .f2k) or as Excel / "
            "Access tables and import that file.")
    out_dir = out_dir or default_out_dir(path)
    os.makedirs(out_dir, exist_ok=True)
    # what is in the folder before this run (path -> (mtime, size)): only files this run
    # writes or rewrites are its output - an older file of the same name is never taken
    before = {}
    for p in glob.glob(os.path.join(out_dir, "*")):
        try:
            st = os.stat(p)
            before[p] = (st.st_mtime_ns, st.st_size)
        except OSError:
            pass
    cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", SCRIPT, "-OutDir", out_dir,
           os.path.abspath(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(f"Conversion of {path} timed out after {timeout:.0f} s") from exc
    log = (proc.stdout or "") + (proc.stderr or "")
    base = os.path.splitext(os.path.basename(path))[0]

    def fresh(p):
        try:
            st = os.stat(p)
        except OSError:
            return False
        return p not in before or before[p] != (st.st_mtime_ns, st.st_size)

    # the converter writes <name>.e2k (ETABS only) and <name>.<Program>.tables.txt
    etabs = os.path.splitext(path)[1].lower() in (".edb",)
    produced = [p for p in glob.glob(os.path.join(out_dir, glob.escape(base) + ".*"))
                if (p.lower().endswith(".tables.txt") or (etabs and p.lower().endswith(".e2k"))) and fresh(p)]
    if proc.returncode != 0 or not produced:
        raise ConversionError(f"The installed CSI program could not convert {os.path.basename(path)}:\n{log.strip()}")
    return sorted(produced, key=lambda p: (not p.lower().endswith(".e2k"), p))

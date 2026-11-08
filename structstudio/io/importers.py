"""One entry point for every model file of ETABS, SAP2000 and SAFE.

``import_model(path)`` recognises the file by extension *and* content:

==========================  =================================================
ETABS                       ``.e2k`` / ``.$et`` text (v9 - 22), database-table
                            exports (``.xlsx``, ``.mdb``, ``.accdb``, ``.txt``),
                            native ``.edb`` (converted by the installed ETABS)
SAP2000 / CSiBridge         ``.s2k`` / ``.$2k`` tables (v8 - 26), legacy
                            keyword ``.s2k`` (v6 - v7, beta), table exports,
                            native ``.sdb`` (converted by the installed SAP2000)
SAFE                        ``.f2k`` / ``.$f2`` tables (v12 - 2016), SAFE 20+
                            table exports, native ``.fdb`` (converted by SAFE)
==========================  =================================================

Native binary files are proprietary; they are opened by the CSI program
installed on the computer (through its official API, see
``scripts/Convert-CsiModels.ps1``), which writes them as text first.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from ..model import Model

NATIVE = {".edb": "ETABS", ".sdb": "SAP2000", ".fdb": "SAFE", ".bdb": "CSiBridge"}
TEXT_MODEL = (".e2k", ".$et", ".s2k", ".$2k", ".f2k", ".$f2", ".b2k", ".$br", ".txt", ".tables")
TABLE_FILES = (".xlsx", ".xlsm", ".xls", ".mdb", ".accdb")

FILE_FILTER = ("CSI models (*.e2k *.$et *.edb *.s2k *.$2k *.sdb *.f2k *.$f2 *.fdb *.b2k *.xlsx *.xls *.mdb *.accdb "
               "*.txt);;ETABS (*.e2k *.$et *.edb *.xlsx *.mdb *.accdb *.txt);;"
               "SAP2000 (*.s2k *.$2k *.sdb *.xlsx *.mdb *.accdb *.txt);;SAFE (*.f2k *.$f2 *.fdb *.xlsx *.mdb *.accdb *.txt);;"
               "All files (*)")


@dataclass
class ImportResult:
    model: Model
    kind: str  # e2k | tables | s2k_legacy
    program: str = ""
    version: str = ""
    path: str = ""
    converted_from: Optional[str] = None  # native file converted by the CSI program
    warnings: List[str] = field(default_factory=list)
    skipped: dict = field(default_factory=dict)
    report: str = ""


def _head(path: str, size: int = 65536) -> str:
    with open(path, "rb") as fh:
        raw = fh.read(size)
    for enc in ("utf-8", "utf-16", "cp1252"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text.lstrip("﻿")
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def detect(path: str) -> str:
    """``native`` | ``tables`` | ``e2k`` | ``s2k_legacy`` | ``unknown``."""
    ext = os.path.splitext(path)[1].lower()
    if ext in NATIVE:
        return "native"
    if ext in TABLE_FILES:
        return "tables"
    try:
        head = _head(path)
    except OSError:
        return "unknown"
    if "\x00" in head[:4096]:
        return "native" if ext not in TEXT_MODEL else "unknown"
    if re.search(r"^\s*TABLE:\s", head, re.MULTILINE):
        return "tables"
    if re.search(r"^\s*\$\s*(PROGRAM INFORMATION|CONTROLS|STORIES|POINT COORDINATES|MATERIAL)", head,
                 re.MULTILINE | re.IGNORECASE):
        return "e2k"
    if re.search(r"^\s*(SYSTEM|JOINTS?|FRAME|SHELL|RESTRAINTS?)\s*$", head, re.MULTILINE | re.IGNORECASE):
        return "s2k_legacy"
    return "unknown"


def program_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in NATIVE:
        return NATIVE[ext]
    return {".e2k": "ETABS", ".$et": "ETABS", ".s2k": "SAP2000", ".$2k": "SAP2000", ".f2k": "SAFE",
            ".$f2": "SAFE", ".b2k": "CSiBridge"}.get(ext, "")


def import_model(path: str, convert_native: bool = True, out_dir: Optional[str] = None,
                 timeout: float = 900.0) -> ImportResult:
    """Import any ETABS / SAP2000 / SAFE model file (see module docs)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    kind = detect(path)
    if kind == "native":
        if not convert_native:
            raise ValueError(f"{os.path.basename(path)} is a native {program_of(path) or 'CSI'} file; "
                             "export it to text first or allow conversion with the installed program")
        from .native import convert

        produced = convert(path, out_dir=out_dir, timeout=timeout)
        preferred = [p for p in produced if p.lower().endswith(".e2k")] + \
                    [p for p in produced if p.lower().endswith(".tables.txt")]
        last_exc: Optional[Exception] = None
        for p in preferred:
            try:
                res = import_model(p, convert_native=False)
                res.converted_from = path
                res.report = f"Converted {os.path.basename(path)} with the installed {program_of(path)}\n" + res.report
                return res
            except Exception as exc:  # try the next file the converter wrote
                last_exc = exc
        raise RuntimeError(f"Conversion of {path} produced no importable file") from last_exc
    if kind == "e2k":
        from .e2k import E2KImporter

        imp = E2KImporter.read(path)
        prog = "ETABS"
        return ImportResult(imp.model, "e2k", prog, "", path, None, list(imp.warnings), dict(imp.skipped), imp.report())
    if kind == "tables":
        from .csi_import import CsiImporter

        if os.path.splitext(path)[1].lower() == ".xls":
            raise ValueError("Excel 97-2003 (.xls) files are not supported; save the workbook as .xlsx "
                             "or export the tables from the CSI program as .xlsx / text")
        imp = CsiImporter.read(path)
        return ImportResult(imp.model, "tables", imp.program, imp.version, path, None, list(imp.warnings),
                            dict(imp.skipped), imp.report())
    if kind == "s2k_legacy":
        from .s2k_legacy import LegacyS2KImporter

        imp = LegacyS2KImporter.read(path)
        return ImportResult(imp.model, "s2k_legacy", "SAP2000", "v6-v7", path, None, list(imp.warnings),
                            dict(imp.skipped), imp.report())
    raise ValueError(f"{os.path.basename(path)}: not a recognised ETABS / SAP2000 / SAFE model file")


def main(argv=None):  # pragma: no cover - command line helper
    import argparse

    ap = argparse.ArgumentParser(description="Import ETABS / SAP2000 / SAFE model files")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--save", action="store_true", help="save each model as <name>.ssj.json")
    a = ap.parse_args(argv)
    for f in a.files:
        res = import_model(f)
        print(res.report)
        if a.save:
            out = os.path.splitext(f)[0] + ".ssj.json"
            res.model.to_json(out)
            print("saved", out)


if __name__ == "__main__":  # pragma: no cover
    main()

"""CSI database tables: the common exchange format of SAP2000, ETABS and SAFE.

All three programs can write their model as a set of *database tables* –
named tables of records with named fields.  The same tables come in several
containers, all read here into one :class:`TableSet`:

* **text** – SAP2000 ``.s2k`` / ``.$2k`` (v8 and later), SAFE ``.f2k`` (v12 and
  later), the "Export > Tables to Text" files of ETABS, and the ``*.tables.txt``
  written by ``scripts/Convert-CsiModels.ps1``::

      TABLE:  "JOINT COORDINATES"
         Joint=1   CoordSys=GLOBAL   CoordType=Cartesian   XorR=0   Y=0   Z=0
         Area=1   NumJoints=5   Joint1=1   Joint2=2 _
            Joint3=3   Joint4=4   Joint5=5          (" _" continues a record)

* **Excel** ``.xlsx`` – one sheet per table: row 1 ``TABLE:  name``, row 2 the
  field names, row 3 the units, then the records.
* **Access** ``.mdb`` / ``.accdb`` – one database table per table (read with
  ``pyodbc`` and the Microsoft Access driver, or ``mdbtools`` where installed).

Table and field names differ between programs and versions ("UniqueName" vs
"Unique Name", "LoadCase" vs "LoadPat") so every name is *normalised*
(lower case, alphanumerics only) and the model builder
(:mod:`structstudio.io.csi_import`) looks fields up through alias lists.  When
an installed ETABS / SAFE provides ``Table and Field Keys.xml`` its display
names are mapped to the field keys as well.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

_NORM = re.compile(r"[^a-z0-9]")
_FIELD = re.compile(r'([^\s=]+)=("([^"]*)"?|\S*)')


def norm(text) -> str:
    """Normalised table / field name: lower case, letters and digits only."""
    return _NORM.sub("", str(text).lower())


# ----------------------------------------------------------------------------
# values
# ----------------------------------------------------------------------------
def to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    t = str(value).strip()
    if not t:
        return default
    try:
        return float(t)
    except ValueError:
        pass
    # "10 (Centroid)", "1,5" (decimal comma), "1.2E+03 kN"
    m = re.match(r"^[-+]?(\d+[.,]?\d*|[.,]\d+)([eE][-+]?\d+)?", t)
    if m:
        try:
            return float(m.group(0).replace(",", "."))
        except ValueError:
            return default
    return default


def to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    t = str(value).strip().lower()
    if t in ("yes", "true", "y", "1", "-1", "on"):
        return True
    if t in ("no", "false", "n", "0", "off", ""):
        return False if t else default
    return default


# ----------------------------------------------------------------------------
# containers
# ----------------------------------------------------------------------------
@dataclass
class Table:
    """One database table.  ``rows`` map *normalised* field names to raw values."""

    name: str
    fields: List[str] = field(default_factory=list)  # original field names in order
    rows: List[Dict[str, object]] = field(default_factory=list)
    units: Dict[str, str] = field(default_factory=dict)  # normalised field -> unit label

    @property
    def key(self) -> str:
        return norm(self.name)

    def has(self, *aliases: str) -> bool:
        keys = {norm(f) for f in self.fields} | {k for r in self.rows[:3] for k in r}
        return any(norm(a) in keys for a in aliases)

    def __len__(self):
        return len(self.rows)

    def __iter__(self) -> Iterator[Dict[str, object]]:
        return iter(self.rows)


def get(row: Dict[str, object], *aliases: str, default=None):
    """First present (non-empty) value of ``row`` among ``aliases``."""
    for a in aliases:
        v = row.get(norm(a))
        if v is not None and v != "":
            return v
    return default


def getf(row, *aliases, default: Optional[float] = None) -> Optional[float]:
    return to_float(get(row, *aliases), default)


def gets(row, *aliases, default: Optional[str] = None) -> Optional[str]:
    v = get(row, *aliases)
    if v is None:
        return default
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


@dataclass
class TableSet:
    """All tables of one model file."""

    tables: Dict[str, Table] = field(default_factory=dict)
    source: str = ""
    container: str = ""  # text | xlsx | access | csv

    def add(self, table: Table) -> Table:
        k = table.key
        if k in self.tables:  # same table split over several blocks / sheets
            old = self.tables[k]
            for f in table.fields:
                if f not in old.fields:
                    old.fields.append(f)
            old.rows += table.rows
            old.units.update(table.units)
            return old
        self.tables[k] = table
        return table

    def __contains__(self, name) -> bool:
        return norm(name) in self.tables

    def __len__(self):
        return len(self.tables)

    def get(self, *names: str) -> Optional[Table]:
        for n in names:
            t = self.tables.get(norm(n))
            if t is not None:
                return t
        return None

    def rows(self, *names: str) -> List[Dict[str, object]]:
        """Records of the first existing table among ``names``."""
        t = self.get(*names)
        return t.rows if t is not None else []

    def matching(self, *prefixes: str, contains: Sequence[str] = ()) -> List[Table]:
        """Tables whose normalised name starts with one of ``prefixes`` (and contains
        every string in ``contains``)."""
        ps = [norm(p) for p in prefixes]
        cs = [norm(c) for c in contains]
        return [t for k, t in self.tables.items() if any(k.startswith(p) for p in ps) and all(c in k for c in cs)]

    def names(self) -> List[str]:
        return [t.name for t in self.tables.values()]

    # ------------------------------------------------------------ program
    def program_control(self) -> Dict[str, object]:
        rows = self.rows("PROGRAM CONTROL", "Program Control")
        return rows[0] if rows else {}

    def apply_field_keys(self, keys: "FieldKeys") -> int:
        """Rename display field names (e.g. "Section Property") to field keys
        ("SectProp") using an ETABS / SAFE ``Table and Field Keys.xml``."""
        renamed = 0
        for t in list(self.tables.values()):
            info = keys.lookup(t.name)
            if info is None:
                continue
            tkey, fmap = info
            for row in t.rows:
                for disp, fkey in fmap.items():
                    if disp in row and fkey not in row:
                        row[fkey] = row.pop(disp)
                        renamed += 1
            if norm(tkey) != t.key:
                del self.tables[t.key]
                t.name = tkey
                self.add(t)
        return renamed


# ----------------------------------------------------------------------------
# Table and Field Keys.xml (ETABS / SAFE)
# ----------------------------------------------------------------------------
class FieldKeys:
    """Display name -> key maps from CSI ``Table and Field Keys.xml`` files."""

    def __init__(self):
        self.tables: Dict[str, Tuple[str, Dict[str, str]]] = {}

    def load(self, path: str) -> "FieldKeys":
        root = ET.parse(path).getroot()
        ns = {"c": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        pre = "c:" if ns else ""
        for t in root.findall(f"{pre}table", ns):
            tkey = (t.findtext(f"{pre}tkey", default="", namespaces=ns) or "").strip()
            if not tkey:
                continue
            fmap = {}
            for f in t.findall(f"{pre}field", ns):
                fk = norm(f.findtext(f"{pre}fkey", default="", namespaces=ns))
                fn = norm(f.findtext(f"{pre}fname", default="", namespaces=ns))
                if fk and fn and fk != fn:
                    fmap[fn] = fk
            for name in (tkey, t.findtext(f"{pre}tname", default="", namespaces=ns),
                         t.findtext(f"{pre}excelname", default="", namespaces=ns)):
                if name:
                    self.tables.setdefault(norm(name), (tkey, fmap))
        return self

    def lookup(self, table_name: str) -> Optional[Tuple[str, Dict[str, str]]]:
        return self.tables.get(norm(table_name))

    @classmethod
    def installed(cls) -> Optional["FieldKeys"]:
        """Keys of the newest ETABS / SAFE installed, if any (Windows)."""
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramW6432", "")]
        found = []
        for r in roots:
            if r:
                found += glob.glob(os.path.join(r, "Computers and Structures", "*", "Table and Field Keys.xml"))
        if not found:
            return None
        keys = cls()
        for p in sorted(found, reverse=True):
            try:
                keys.load(p)
            except (ET.ParseError, OSError):
                continue
        return keys if keys.tables else None


# ----------------------------------------------------------------------------
# text tables
# ----------------------------------------------------------------------------
def looks_like_text_tables(text: str) -> bool:
    head = text[:20000]
    return bool(re.search(r"^\s*TABLE:\s", head, re.MULTILINE))


def _parse_record(line: str) -> List[Tuple[str, str]]:
    out = []
    for m in _FIELD.finditer(line):
        name = m.group(1)
        val = m.group(3) if m.group(2).startswith('"') else m.group(2)
        out.append((name, val if val is not None else ""))
    return out


def read_text_tables(text: str, source: str = "") -> TableSet:
    ts = TableSet(source=source, container="text")
    current: Optional[Table] = None
    pending = ""
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if pending:
            line = pending + " " + stripped
            stripped = line.strip()
            pending = ""
        if not stripped:
            continue
        if stripped.endswith(" _") or stripped == "_":
            pending = stripped[:-1].rstrip()
            continue
        up = stripped.upper()
        if up.startswith("TABLE:"):
            name = stripped[6:].strip().strip('"').strip()
            current = ts.add(Table(name=name))
            continue
        if up.startswith("END TABLE DATA"):
            break
        if current is None or "=" not in stripped:
            continue  # file header ("File x.s2k was saved on ...") or comments
        rec = _parse_record(stripped)
        if not rec:
            continue
        row = {}
        for name, val in rec:
            if name not in current.fields:
                current.fields.append(name)
            row[norm(name)] = val
        current.rows.append(row)
    if pending and current is not None:
        rec = _parse_record(pending)
        if rec:
            current.rows.append({norm(n): v for n, v in rec})
    return ts


def write_text_tables(ts: TableSet, program: str = "") -> str:
    """Write tables back in the CSI text format (used by tests and the converter)."""
    out = []
    for t in ts.tables.values():
        out.append(f'TABLE:  "{t.name}"')
        names = t.fields or sorted({k for r in t.rows for k in r})
        for row in t.rows:
            parts = []
            for f in names:
                v = row.get(norm(f))
                if v is None:
                    continue
                v = str(v)
                if v == "" or any(c.isspace() for c in v) or '"' in v:
                    v = '"' + v.replace('"', "'") + '"'
                parts.append(f"{f}={v}")
            out.append("   " + "   ".join(parts))
        out.append("")
    out.append('END TABLE DATA')
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------
# Excel
# ----------------------------------------------------------------------------
def _cell_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return repr(v) if not v.is_integer() else str(int(v)) if abs(v) < 1e15 else repr(v)
    return str(v).strip()


def read_xlsx_tables(path: str) -> TableSet:
    try:
        import openpyxl
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Reading Excel tables needs openpyxl (pip install openpyxl)") from exc
    ts = TableSet(source=path, container="xlsx")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for ws in wb.worksheets:
            rows = ws.iter_rows(values_only=True)
            first = next(rows, None)
            if first is None:
                continue
            title = _cell_str(first[0]) if first else ""
            if title.upper().startswith("TABLE:"):
                name = title[6:].strip().strip('"')
                header = next(rows, None)
            else:
                name = ws.title
                header = first
            if not header:
                continue
            fields = [_cell_str(h) for h in header]
            if not any(fields):
                continue
            t = Table(name=name, fields=[f for f in fields if f])
            data = list(rows)
            # units row: strings only, at least one unit-like label, before the records
            if data:
                u = [_cell_str(c) for c in data[0]]
                numeric = [to_float(c) is not None for c in data[0] if c not in (None, "")]
                if not any(numeric) and len(data) > 1 and any(u):
                    t.units = {norm(f): unit for f, unit in zip(fields, u) if f and unit}
                    data = data[1:]
            for rec in data:
                if rec is None or all(c is None or c == "" for c in rec):
                    continue
                row = {norm(f): _cell_str(c) for f, c in zip(fields, rec) if f and c is not None and c != ""}
                if row:
                    t.rows.append(row)
            ts.add(t)
    finally:
        wb.close()
    return ts


# ----------------------------------------------------------------------------
# Access
# ----------------------------------------------------------------------------
def read_access_tables(path: str) -> TableSet:
    """Tables of a CSI Access export.  Uses pyodbc + the Microsoft Access driver
    (Windows) or mdbtools (``mdb-tables`` / ``mdb-export``) when available."""
    ts = TableSet(source=path, container="access")
    try:
        import pyodbc  # type: ignore
        drivers = [d for d in pyodbc.drivers() if "access" in d.lower()]
    except ImportError:
        pyodbc, drivers = None, []
    if pyodbc is not None and drivers:
        conn = pyodbc.connect(f"DRIVER={{{drivers[0]}}};DBQ={os.path.abspath(path)};", autocommit=True)
        try:
            cur = conn.cursor()
            names = [r.table_name for r in cur.tables(tableType="TABLE")]
            for name in names:
                cur.execute(f"SELECT * FROM [{name}]")
                fields = [d[0] for d in cur.description]
                t = Table(name=name, fields=fields)
                for rec in cur.fetchall():
                    row = {norm(f): _cell_str(v) for f, v in zip(fields, rec) if v is not None and v != ""}
                    if row:
                        t.rows.append(row)
                ts.add(t)
        finally:
            conn.close()
        return ts
    if shutil.which("mdb-tables") and shutil.which("mdb-export"):
        import csv
        import io

        names = subprocess.run(["mdb-tables", "-1", path], capture_output=True, text=True, check=True).stdout.splitlines()
        for name in names:
            if not name.strip():
                continue
            out = subprocess.run(["mdb-export", path, name], capture_output=True, text=True, check=True).stdout
            reader = csv.reader(io.StringIO(out))
            fields = next(reader, [])
            t = Table(name=name, fields=fields)
            for rec in reader:
                row = {norm(f): v for f, v in zip(fields, rec) if v != ""}
                if row:
                    t.rows.append(row)
            ts.add(t)
        return ts
    raise ImportError(
        "Reading an Access (.mdb / .accdb) export needs pyodbc with the Microsoft Access Database Engine "
        "(64-bit) driver, or mdbtools.  Alternatively export the tables from the CSI program to Excel "
        "(.xlsx) or text (.s2k / .f2k / .txt) and import that file.")


# ----------------------------------------------------------------------------
def read_tables(path: str, field_keys: Optional[FieldKeys] = None) -> TableSet:
    """Read any table container by extension / content."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm"):
        ts = read_xlsx_tables(path)
    elif ext in (".mdb", ".accdb"):
        ts = read_access_tables(path)
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        if text.startswith("\ufeff"):
            text = text[1:]
        ts = read_text_tables(text, source=path)
    keys = field_keys if field_keys is not None else FieldKeys.installed()
    if keys is not None:
        ts.apply_field_keys(keys)
    return ts


def iter_named(tables: Iterable[Table], *id_fields: str) -> Iterator[Tuple[str, Dict[str, object], Table]]:
    """(id, row, table) over the records of ``tables`` that have one of ``id_fields``."""
    for t in tables:
        for row in t.rows:
            ident = gets(row, *id_fields)
            if ident:
                yield ident, row, t

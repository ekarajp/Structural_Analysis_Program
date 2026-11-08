"""Command line interface.

    structstudio info   MODEL            summary of a model (.json or .e2k/.$et)
    structstudio import MODEL.e2k -o out.json
    structstudio analyze MODEL [--cases A B] [-o results.json]
    structstudio verify                  run the CSI verification suite
"""
from __future__ import annotations

import argparse
import json
import sys

from . import analyze, __version__
from .io import E2KImporter
from .model import Model, ModelError


def load_model(path: str) -> Model:
    low = path.lower()
    if low.endswith((".e2k", ".$et", ".txt")):
        imp = E2KImporter.read(path)
        if imp.warnings or imp.skipped:
            print(imp.report(), file=sys.stderr)
        return imp.model
    return Model.from_json(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="structstudio", description="Structural Studio engine")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="print a model summary")
    p.add_argument("model")

    p = sub.add_parser("import", help="convert an ETABS text file to JSON")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True)

    p = sub.add_parser("analyze", help="run load cases")
    p.add_argument("model")
    p.add_argument("--cases", nargs="*")
    p.add_argument("-o", "--output", help="write results JSON")
    p.add_argument("--frames", nargs="*", help="print station forces for these frames")

    sub.add_parser("verify", help="run the verification test-suite (needs pytest)")

    args = ap.parse_args(argv)
    try:
        if args.cmd == "info":
            m = load_model(args.model)
            print(json.dumps(m.summary(), indent=1, ensure_ascii=False))
            errors = m.validate()
            for e in errors:
                print(f"ERROR {e.code}: {e.message}", file=sys.stderr)
            return 1 if errors else 0
        if args.cmd == "import":
            m = load_model(args.model)
            m.to_json(args.output)
            print(f"wrote {args.output}: {m!r}")
            return 0
        if args.cmd == "analyze":
            m = load_model(args.model)
            res = analyze(m, cases=args.cases)
            for name, r in res.cases.items():
                line = f"{name:20s} {r.type:18s}"
                if r.modal is not None and r.type == "Modal":
                    line += " periods: " + ", ".join(f"{t:.4f}" for t in r.modal.periods[:6])
                else:
                    br = res.base_reactions(name)
                    line += f" base reactions FX={br[0]:.4g} FY={br[1]:.4g} FZ={br[2]:.4g}"
                print(line)
                for w in r.warnings:
                    print(f"   warning: {w}")
            for w in res.am.warnings:
                print(f"warning: {w}")
            if args.frames:
                for f in args.frames:
                    for name in res.cases:
                        if res.cases[name].type == "Modal":
                            continue
                        st = res.frame_forces(f, name)
                        print(f"\n{f} / {name}")
                        print(f"{'Station':>10s} {'P':>12s} {'V2':>12s} {'V3':>12s} {'T':>12s} {'M2':>12s} {'M3':>12s}")
                        for k in range(len(st.station)):
                            print(f"{st.station[k]:10.3f} " + " ".join(f"{v:12.4f}" for v in st.forces[k]))
            if args.output:
                with open(args.output, "w", encoding="utf-8") as fh:
                    json.dump(res.to_dict(), fh, indent=1, ensure_ascii=False)
                print(f"wrote {args.output}")
            return 0
        if args.cmd == "verify":
            import os
            import pytest

            tests = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
            return pytest.main(["-q", tests])
    except ModelError as e:
        print(f"ERROR {e.code}: {e.message}" + (f" ({e.path})" if e.path else ""), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

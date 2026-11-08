"""Engineering text as it is printed: powers as superscripts (m², cm⁴), section
and material symbols with subscripts (A<sub>s2</sub>, I<sub>33</sub>, f′<sub>c</sub>,
f<sub>y</sub>), Greek letter names as letters (φ, γ, α ...).

``pretty`` gives plain Unicode (table cells, combo boxes, the model window);
``rich`` gives Qt rich text (labels)."""
from __future__ import annotations

import re

_SUP = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")
_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
GREEK = {"phi": "φ", "Phi": "Φ", "alpha": "α", "beta": "β", "gamma": "γ", "Gamma": "Γ", "delta": "δ",
         "Delta": "Δ", "epsilon": "ε", "theta": "θ", "lambda": "λ", "mu": "μ", "nu": "ν", "rho": "ρ",
         "sigma": "σ", "Sigma": "Σ", "tau": "τ", "omega": "ω", "Omega": "Ω", "xi": "ξ", "eta": "η"}
_GREEK_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(GREEK, key=len, reverse=True)) + r")(?![A-Za-z])")
_POW_RE = re.compile(r"\^(-?\d+)")
# section / material symbols: letter + subscript (whole words only)
_SYM_RE = re.compile(r"(?<![A-Za-z0-9])(As|I|S|Z|r|M|V|F|U|R)(1|2|3|11|22|33|12|13|23|23)(?![A-Za-z0-9])")
_FC_RE = re.compile(r"(?<![A-Za-z])(?:fc'|f'c|f′c)(?![A-Za-z])")
_F_RE = re.compile(r"(?<![A-Za-z])f(y|u|ye|ue|ys)(?![A-Za-z])")
_E_RE = re.compile(r"(?<![A-Za-z])E(c|s)(?![A-Za-z])")


def pretty_unit(label: str) -> str:
    """``m^2`` -> ``m²``, ``kN/m^3`` -> ``kN/m³``, ``cm^4`` -> ``cm⁴``."""
    return _POW_RE.sub(lambda m: m.group(1).translate(_SUP), str(label))


def plain_unit(label: str) -> str:
    """The inverse: superscripts back to ``^n`` (for parsing)."""
    s = str(label)
    inv = {v: k for k, v in zip("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")}
    return re.sub(r"[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+", lambda m: "^" + "".join(inv[c] for c in m.group(0)), s)


def pretty(text: str) -> str:
    """Plain Unicode: powers, Greek names and the digit subscripts of section symbols."""
    s = pretty_unit(text)
    s = _GREEK_RE.sub(lambda m: GREEK[m.group(1)], s)
    s = _FC_RE.sub("f′c", s)
    s = _SYM_RE.sub(lambda m: m.group(1).replace("As", "Aₛ") + m.group(2).translate(_SUB), s)
    return s


def rich(text: str) -> str:
    """Qt rich text: like :func:`pretty` with real subscripts (f′<sub>c</sub>, f<sub>y</sub>,
    E<sub>c</sub>, A<sub>s2</sub>)."""
    s = pretty_unit(str(text)).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = _GREEK_RE.sub(lambda m: GREEK[m.group(1)], s)
    s = _FC_RE.sub("f′<sub>c</sub>", s)
    s = _F_RE.sub(lambda m: f"f<sub>{m.group(1)}</sub>", s)
    s = _E_RE.sub(lambda m: f"E<sub>{m.group(1)}</sub>", s)
    s = _SYM_RE.sub(lambda m: ("A<sub>s" if m.group(1) == "As" else m.group(1) + "<sub>")
                    + m.group(2) + "</sub>", s)
    return s
